/**
 * 推荐引擎：论文 6.3 推荐服务层
 * - 场景化调度（6.3.2）：cold_start→SoCoGNN / noisy_social→SoDRA / knowledge_rich→PESatNet / normal→三模型融合
 * - 加权融合（6.3.3）：三模型 64 维嵌入得分经 sigmoid 归一后按场景权重加权
 * - 服务层策略：8 条利用 + 2 条长尾探索（解决长尾零曝光问题，UI 标注"长尾探索"）
 */
import { getDb } from "./connection";
import { embeddings, items, modelStats } from "../../db/schema";
import { eq } from "drizzle-orm";

export const MODELS = ["socognn", "sodra", "pesatnet"] as const;
export type ModelName = (typeof MODELS)[number];
export const MODEL_LABEL: Record<ModelName, string> = {
  socognn: "SoCoGNN",
  sodra: "SoDRA",
  pesatnet: "PESatNet",
};

type Quant = { s: number; q: number[] };

interface Store {
  userEmb: Record<ModelName, Float32Array[]>; // [userIdx][dim]
  itemEmb: Record<ModelName, Float32Array[]>;
  userHistory: Map<number, Set<number>>;
  sceneWeights: Record<string, [number, number, number]>;
  sceneAlgo: Record<string, string>;
  longTailIds: number[];
  itemPop: Map<number, number>;
}

let store: Store | null = null;

function dequant(q: Quant): Float32Array {
  const out = new Float32Array(q.q.length);
  for (let i = 0; i < q.q.length; i++) out[i] = q.q[i] * q.s;
  return out;
}

async function loadStore(): Promise<Store> {
  if (store) return store;
  const db = getDb();
  const rows = await db.select().from(embeddings);
  const NU = 10000;
  const userEmb = { socognn: new Array(NU), sodra: new Array(NU), pesatnet: new Array(NU) } as Store["userEmb"];
  const itemEmb = { socognn: [], sodra: [], pesatnet: [] } as unknown as Store["itemEmb"];
  for (const r of rows) {
    const target = r.entityType === "user" ? userEmb : itemEmb;
    for (const m of MODELS) {
      (target[m] as Float32Array[])[r.entityId] = dequant(JSON.parse(r[m]));
    }
  }
  const histRow = await db.select().from(modelStats).where(eq(modelStats.key, "user_history"));
  const hist = JSON.parse(histRow[0].payload) as Record<string, number[]>;
  const userHistory = new Map<number, Set<number>>();
  for (const [u, its] of Object.entries(hist)) userHistory.set(Number(u), new Set(its));

  const swRow = await db.select().from(modelStats).where(eq(modelStats.key, "scene_weights"));
  const saRow = await db.select().from(modelStats).where(eq(modelStats.key, "scene_algo"));
  const itemRows = await db
    .select({ itemId: items.itemId, isLongTail: items.isLongTail, popularity: items.popularity })
    .from(items);
  store = {
    userEmb,
    itemEmb,
    userHistory,
    sceneWeights: JSON.parse(swRow[0].payload),
    sceneAlgo: JSON.parse(saRow[0].payload),
    longTailIds: itemRows.filter((i) => i.isLongTail === 1).map((i) => i.itemId),
    itemPop: new Map(itemRows.map((i) => [i.itemId, i.popularity])),
  };
  return store;
}

const sigmoid = (x: number) => 1 / (1 + Math.exp(-x));

export interface ScoredItem {
  itemId: number;
  score: number; // 融合分
  perModel: Record<ModelName, number>; // 各模型 sigmoid 分
  source: string; // 主导算法
  slot: "exploit" | "explore";
}

/** 论文 6.3.3 加权融合 + 8 利用 / 2 长尾探索槽位 */
export async function scoreForUser(
  userId: number,
  scene: string,
  k = 10,
): Promise<{ scene: string; weights: [number, number, number]; mainAlgo: string; recs: ScoredItem[] }> {
  const s = await loadStore();
  const weights = s.sceneWeights[scene] ?? s.sceneWeights.normal;
  const mainAlgo = s.sceneAlgo[scene] ?? "Fusion";
  const seen = s.userHistory.get(userId) ?? new Set<number>();
  const NI = s.itemEmb.socognn.length;

  const scores = new Float64Array(NI);
  const perModel: [Float64Array, Float64Array, Float64Array] = [
    new Float64Array(NI),
    new Float64Array(NI),
    new Float64Array(NI),
  ];
  for (let m = 0; m < 3; m++) {
    const uv = s.userEmb[MODELS[m]][userId];
    const arr = perModel[m];
    // 原始点积
    for (let i = 0; i < NI; i++) {
      const iv = s.itemEmb[MODELS[m]][i];
      let dot = 0;
      for (let d = 0; d < 64; d++) dot += uv[d] * iv[d];
      arr[i] = dot;
    }
    // 按论文 6.3.3 对各模型得分做归一（z-score → sigmoid）后再加权融合
    // 点积分布右尾重（好物品 z≈10+），除以 4 压缩量程避免 sigmoid 饱和
    let mean = 0;
    for (let i = 0; i < NI; i++) mean += arr[i];
    mean /= NI;
    let varSum = 0;
    for (let i = 0; i < NI; i++) varSum += (arr[i] - mean) ** 2;
    const std = Math.sqrt(varSum / NI) || 1e-9;
    for (let i = 0; i < NI; i++) arr[i] = sigmoid(((arr[i] - mean) / std) / 4);
  }
  for (let i = 0; i < NI; i++) {
    scores[i] =
      weights[0] * perModel[0][i] + weights[1] * perModel[1][i] + weights[2] * perModel[2][i];
  }

  const candidates: number[] = [];
  for (let i = 0; i < NI; i++) if (!seen.has(i)) candidates.push(i);
  candidates.sort((a, b) => scores[b] - scores[a]);

  const mkRec = (itemId: number, slot: "exploit" | "explore"): ScoredItem => {
    const pm: Record<ModelName, number> = {
      socognn: perModel[0][itemId],
      sodra: perModel[1][itemId],
      pesatnet: perModel[2][itemId],
    };
    let src: ModelName = "socognn";
    let best = -1;
    for (let m = 0; m < 3; m++) {
      const contrib = weights[m] * perModel[m][itemId];
      if (contrib > best) {
        best = contrib;
        src = MODELS[m];
      }
    }
    return { itemId, score: scores[itemId], perModel: pm, source: MODEL_LABEL[src], slot };
  };

  const nExploit = Math.max(k - 2, 1);
  const nExplore = k - nExploit;
  const exploit = candidates.slice(0, nExploit).map((id) => mkRec(id, "exploit"));
  const picked = new Set(exploit.map((r) => r.itemId));
  const explore = s.longTailIds
    .filter((id) => !seen.has(id) && !picked.has(id))
    .sort((a, b) => scores[b] - scores[a])
    .slice(0, nExplore)
    .map((id) => mkRec(id, "explore"));

  return { scene, weights, mainAlgo, recs: [...exploit, ...explore] };
}
