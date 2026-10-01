import { z } from "zod";
import { and, desc, eq, inArray, like, or, sql } from "drizzle-orm";
import { createRouter, publicQuery } from "./middleware";
import { getDb } from "./queries/connection";
import { scoreForUser, MODEL_LABEL, type ModelName } from "./queries/reco";
import {
  users,
  items,
  interactions,
  socialEdges,
  kgEntities,
  kgTriples,
  modelStats,
  feedback,
} from "../db/schema";

async function getStat<T = any>(key: string): Promise<T | null> {
  const db = getDb();
  const rows = await db.select().from(modelStats).where(eq(modelStats.key, key));
  return rows.length ? (JSON.parse(rows[0].payload) as T) : null;
}

const userRouter = createRouter({
  /** 演示用户列表（供切换器选择） */
  list: publicQuery
    .input(
      z.object({
        scene: z.string().optional(),
        userType: z.string().optional(),
        limit: z.number().min(1).max(100).default(24),
        offset: z.number().min(0).default(0),
      }),
    )
    .query(async ({ input }) => {
      const db = getDb();
      const conds = [];
      if (input.scene) conds.push(eq(users.scene, input.scene));
      if (input.userType) conds.push(eq(users.userType, input.userType));
      const where = conds.length ? and(...conds) : undefined;
      const rows = await db
        .select()
        .from(users)
        .where(where)
        .orderBy(users.userId)
        .limit(input.limit)
        .offset(input.offset);
      const cnt = await db
        .select({ c: sql<number>`count(*)` })
        .from(users)
        .where(where);
      return { rows, total: cnt[0].c };
    }),

  /** 用户画像：基本信息 + 行为时间线 + 社交邻居（含 SoDRA 净化权重） */
  profile: publicQuery
    .input(z.object({ userId: z.number() }))
    .query(async ({ input }) => {
      const db = getDb();
      const u = await db.select().from(users).where(eq(users.userId, input.userId));
      if (!u.length) throw new Error("用户不存在");
      const timeline = await db
        .select({
          itemId: interactions.itemId,
          behaviorType: interactions.behaviorType,
          rating: interactions.rating,
          ts: interactions.ts,
          name: items.name,
          category: items.category,
          price: items.price,
        })
        .from(interactions)
        .leftJoin(items, eq(interactions.itemId, items.itemId))
        .where(eq(interactions.userId, input.userId))
        .orderBy(desc(interactions.ts))
        .limit(12);
      const edges = await db
        .select()
        .from(socialEdges)
        .where(or(eq(socialEdges.src, input.userId), eq(socialEdges.dst, input.userId)))
        .limit(50);
      const neighborIds = [...new Set(edges.map((e) => (e.src === input.userId ? e.dst : e.src)))];
      const neighbors = neighborIds.length
        ? await db
            .select({
              userId: users.userId,
              userType: users.userType,
              province: users.province,
              cropType: users.cropType,
            })
            .from(users)
            .where(inArray(users.userId, neighborIds))
        : [];
      const nmap = new Map(neighbors.map((n) => [n.userId, n]));
      const social = edges.map((e) => ({
        ...e,
        neighbor: nmap.get(e.src === input.userId ? e.dst : e.src) ?? null,
      }));
      return { user: u[0], timeline, social };
    }),
});

const recoRouter = createRouter({
  /** 场景化调度推荐（论文 6.3）：8 利用 + 2 长尾探索 */
  recommend: publicQuery
    .input(z.object({ userId: z.number(), k: z.number().min(4).max(30).default(10) }))
    .query(async ({ input }) => {
      const db = getDb();
      const u = await db.select().from(users).where(eq(users.userId, input.userId));
      if (!u.length) throw new Error("用户不存在");
      const result = await scoreForUser(input.userId, u[0].scene, input.k);
      const ids = result.recs.map((r) => r.itemId);
      const itemRows = ids.length
        ? await db.select().from(items).where(inArray(items.itemId, ids))
        : [];
      const imap = new Map(itemRows.map((i) => [i.itemId, i]));
      return {
        user: u[0],
        scene: result.scene,
        mainAlgo: result.mainAlgo,
        weights: {
          socognn: result.weights[0],
          sodra: result.weights[1],
          pesatnet: result.weights[2],
        },
        recs: result.recs.map((r) => ({ ...r, item: imap.get(r.itemId) ?? null })),
      };
    }),

  /** 提交反馈（对应论文 6.1.2 用户反馈收集模块） */
  submitFeedback: publicQuery
    .input(
      z.object({
        userId: z.number(),
        itemId: z.number(),
        action: z.enum(["click", "collect", "purchase", "skip", "like"]),
        scene: z.string().default("normal"),
        algorithm: z.string().default("Fusion"),
      }),
    )
    .mutation(async ({ input }) => {
      const db = getDb();
      await db.insert(feedback).values(input);
      return { ok: true };
    }),

  /** 反馈统计（数据看板） */
  feedbackSummary: publicQuery.query(async () => {
    const db = getDb();
    const rows = await db
      .select({ action: feedback.action, c: sql<number>`count(*)` })
      .from(feedback)
      .groupBy(feedback.action);
    return rows;
  }),
});

const kgRouter = createRouter({
  search: publicQuery
    .input(z.object({ q: z.string().min(1), limit: z.number().min(1).max(50).default(12) }))
    .query(async ({ input }) => {
      const db = getDb();
      return db
        .select()
        .from(kgEntities)
        .where(like(kgEntities.name, `%${input.q}%`))
        .limit(input.limit);
    }),

  /** 实体邻居：一跳三元组（双向） */
  neighbors: publicQuery
    .input(z.object({ entityId: z.number(), limit: z.number().min(1).max(100).default(40) }))
    .query(async ({ input }) => {
      const db = getDb();
      const triples = await db
        .select()
        .from(kgTriples)
        .where(or(eq(kgTriples.h, input.entityId), eq(kgTriples.t, input.entityId)))
        .limit(input.limit);
      const ids = [...new Set(triples.flatMap((t) => [t.h, t.t]))];
      const ents = ids.length
        ? await db.select().from(kgEntities).where(inArray(kgEntities.entityId, ids))
        : [];
      const emap = new Map(ents.map((e) => [e.entityId, e]));
      return triples.map((t) => ({
        h: emap.get(t.h) ?? { entityId: t.h, name: String(t.h), etype: "?" },
        r: t.r,
        t: emap.get(t.t) ?? { entityId: t.t, name: String(t.t), etype: "?" },
      }));
    }),

  /** 物品关联的 KG 子图 */
  itemGraph: publicQuery
    .input(z.object({ itemId: z.number() }))
    .query(async ({ input }) => {
      const db = getDb();
      const it = await db.select().from(items).where(eq(items.itemId, input.itemId));
      if (!it.length || it[0].kgEntityId == null) return { item: it[0] ?? null, triples: [] };
      const eid = it[0].kgEntityId;
      const triples = await db
        .select()
        .from(kgTriples)
        .where(or(eq(kgTriples.h, eid), eq(kgTriples.t, eid)))
        .limit(30);
      const ids = [...new Set(triples.flatMap((t) => [t.h, t.t]))];
      const ents = ids.length
        ? await db.select().from(kgEntities).where(inArray(kgEntities.entityId, ids))
        : [];
      const emap = new Map(ents.map((e) => [e.entityId, e]));
      return {
        item: it[0],
        triples: triples.map((t) => ({
          h: emap.get(t.h) ?? { entityId: t.h, name: String(t.h), etype: "?" },
          r: t.r,
          t: emap.get(t.t) ?? { entityId: t.t, name: String(t.t), etype: "?" },
        })),
      };
    }),

  stats: publicQuery.query(async () => {
    const db = getDb();
    const byType = await db
      .select({ etype: kgEntities.etype, c: sql<number>`count(*)` })
      .from(kgEntities)
      .groupBy(kgEntities.etype);
    const byRel = await db
      .select({ r: kgTriples.r, c: sql<number>`count(*)` })
      .from(kgTriples)
      .groupBy(kgTriples.r)
      .orderBy(desc(sql`count(*)`));
    return { byType, byRel };
  }),
});

const statsRouter = createRouter({
  overview: publicQuery.query(async () => {
    const [metrics, meta, coldCov, ltExp, trainSec] = await Promise.all([
      getStat("metrics"),
      getStat("dataset_meta"),
      getStat<number>("cold_coverage"),
      getStat<number>("long_tail_exposure"),
      getStat<number>("train_seconds"),
    ]);
    return { metrics, meta, coldCoverage: coldCov, longTailExposure: ltExp, trainSeconds: trainSec };
  }),
  curves: publicQuery.query(() => getStat("curves")),
  purify: publicQuery.query(() => getStat("purify")),
  sceneConfig: publicQuery.query(async () => {
    const [weights, algo] = await Promise.all([getStat("scene_weights"), getStat("scene_algo")]);
    return { weights, algo };
  }),
  /** 数据看板分布统计 */
  distributions: publicQuery.query(async () => {
    const db = getDb();
    const [userScenes, userTypes, itemCats, itemLongTail, behaviorTypes] = await Promise.all([
      db.select({ k: users.scene, c: sql<number>`count(*)` }).from(users).groupBy(users.scene),
      db.select({ k: users.userType, c: sql<number>`count(*)` }).from(users).groupBy(users.userType),
      db.select({ k: items.category, c: sql<number>`count(*)` }).from(items).groupBy(items.category),
      db.select({ k: items.isLongTail, c: sql<number>`count(*)` }).from(items).groupBy(items.isLongTail),
      db.select({ k: interactions.behaviorType, c: sql<number>`count(*)` }).from(interactions).groupBy(interactions.behaviorType),
    ]);
    return { userScenes, userTypes, itemCats, itemLongTail, behaviorTypes };
  }),
});

export const appRouter = createRouter({
  ping: publicQuery.query(() => ({ ok: true, ts: Date.now() })),
  user: userRouter,
  reco: recoRouter,
  kg: kgRouter,
  stats: statsRouter,
});

export type AppRouter = typeof appRouter;
export { MODEL_LABEL };
export type { ModelName };
