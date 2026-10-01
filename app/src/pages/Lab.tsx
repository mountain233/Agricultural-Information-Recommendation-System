import { useMemo, useState } from "react";
import { trpc } from "@/providers/trpc";
import { MODEL_META, SCENE_META, KG_TYPE_LABEL, KG_REL_LABEL } from "@/lib/consts";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { Input } from "@/components/ui/input";
import { Skeleton } from "@/components/ui/skeleton";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  BarChart,
  Bar,
} from "recharts";
import { Search } from "lucide-react";

const MODELS = ["socognn", "sodra", "pesatnet"] as const;

/* ---------------- 训练曲线 ---------------- */
function Curves() {
  const q = trpc.stats.curves.useQuery();
  const data = useMemo(() => {
    if (!q.data) return [];
    const byEpoch = new Map<number, Record<string, number>>();
    for (const m of ["SoCoGNN", "SoDRA", "PESatNet"]) {
      for (const e of (q.data as any)[m] ?? []) {
        const row: Record<string, number> = byEpoch.get(e.epoch) ?? { epoch: e.epoch };
        row[`${m}_loss`] = e.loss;
        if (e.val_R20 != null) row[`${m}_r20`] = e.val_R20;
        byEpoch.set(e.epoch, row);
      }
    }
    return [...byEpoch.values()].sort((a, b) => a.epoch - b.epoch);
  }, [q.data]);

  if (q.isLoading) return <Skeleton className="h-72" />;
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
      <div className="panel p-4">
        <div className="panel-title mb-3">TRAINING LOSS · BPR + 正则项</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ left: -18, right: 8, top: 4 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
            <XAxis dataKey="epoch" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} />
            <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} domain={["auto", "auto"]} />
            <Tooltip contentStyle={{ fontSize: 11, fontFamily: "JetBrains Mono", background: "hsl(50 44% 94%)", border: "1px solid hsl(70 16% 74%)" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {["SoCoGNN", "SoDRA", "PESatNet"].map((m) => (
              <Line key={m} dataKey={`${m}_loss`} name={m} dot={false} strokeWidth={1.5}
                stroke={Object.values(MODEL_META).find((x) => x.label === m)!.color} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
      <div className="panel p-4">
        <div className="panel-title mb-3">VALIDATION RECALL@20 · 每 5 EPOCH</div>
        <ResponsiveContainer width="100%" height={260}>
          <LineChart data={data} margin={{ left: -18, right: 8, top: 4 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
            <XAxis dataKey="epoch" tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} />
            <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} domain={[0, 0.8]} />
            <Tooltip contentStyle={{ fontSize: 11, fontFamily: "JetBrains Mono", background: "hsl(50 44% 94%)", border: "1px solid hsl(70 16% 74%)" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            {["SoCoGNN", "SoDRA", "PESatNet"].map((m) => (
              <Line key={m} dataKey={`${m}_r20`} name={m} strokeWidth={1.5} connectNulls
                dot={{ r: 2 }} stroke={Object.values(MODEL_META).find((x) => x.label === m)!.color} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ---------------- 指标对比 ---------------- */
function Metrics() {
  const q = trpc.stats.overview.useQuery();
  if (q.isLoading) return <Skeleton className="h-72" />;
  const metrics = q.data?.metrics as any;
  if (!metrics) return null;
  const subsets = [
    ["overall", "整体测试集"],
    ["cold", "冷启动用户子集"],
    ["noisy", "社交噪声用户子集"],
  ] as const;
  return (
    <div className="space-y-3">
      <div className="grid grid-cols-1 gap-3 md:grid-cols-3">
        {subsets.map(([key, label]) => (
          <div key={key} className="panel p-4">
            <div className="panel-title mb-2">{label}</div>
            <table className="w-full text-xs">
              <thead>
                <tr className="border-b border-border font-mono text-[10px] text-muted-foreground">
                  <th className="py-1 text-left font-normal">MODEL</th>
                  <th className="text-right font-normal">P@10</th>
                  <th className="text-right font-normal">R@20</th>
                  <th className="text-right font-normal">N@10</th>
                </tr>
              </thead>
              <tbody>
                {["SoCoGNN", "SoDRA", "PESatNet", "Fusion"].map((m) => {
                  const s = metrics[m]?.[key];
                  if (!s) return null;
                  const meta = Object.values(MODEL_META).find((x) => x.label === m);
                  return (
                    <tr key={m} className="border-b border-border/50 font-mono">
                      <td className="py-1.5" style={{ color: meta?.color ?? "#40654a" }}>{m}</td>
                      <td className="text-right">{s.P10.toFixed(4)}</td>
                      <td className="text-right font-semibold">{s.R20.toFixed(4)}</td>
                      <td className="text-right">{s.N10.toFixed(4)}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        ))}
      </div>
      <div className="panel p-4">
        <div className="panel-title mb-3">RECALL@20 对比</div>
        <ResponsiveContainer width="100%" height={220}>
          <BarChart
            data={["SoCoGNN", "SoDRA", "PESatNet", "Fusion"].map((m) => ({
              name: m,
              整体: metrics[m].overall.R20,
              冷启动: metrics[m].cold.R20,
              噪声: metrics[m].noisy.R20,
            }))}
            margin={{ left: -18, right: 8, top: 4 }}
          >
            <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
            <XAxis dataKey="name" tick={{ fontSize: 11, fontFamily: "JetBrains Mono" }} />
            <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} domain={[0.6, 0.7]} />
            <Tooltip contentStyle={{ fontSize: 11, fontFamily: "JetBrains Mono", background: "hsl(50 44% 94%)", border: "1px solid hsl(70 16% 74%)" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="整体" fill="#40654a" />
            <Bar dataKey="冷启动" fill="#2f6b3a" />
            <Bar dataKey="噪声" fill="#b98045" />
          </BarChart>
        </ResponsiveContainer>
        <p className="mt-2 font-mono text-[10px] leading-relaxed text-muted-foreground">
          注：三模型在 AgriRec-Sim 上指标接近 —— 合成数据的热度+亲和信号被各架构同等捕获，融合略优。
          算法差异主要体现在机制上（双视图融合 / 边净化 / 知识传播），见其余标签页。
        </p>
      </div>
    </div>
  );
}

/* ---------------- SoDRA 净化可视化 ---------------- */
function Purify() {
  const q = trpc.stats.purify.useQuery();
  if (q.isLoading) return <Skeleton className="h-72" />;
  const p = q.data as any;
  if (!p) return null;
  const bins = p.weight_bins.edges.slice(0, -1).map((e: number, i: number) => ({
    range: e.toFixed(2),
    噪声边: p.weight_bins.noise[i],
    真实边: p.weight_bins.real[i],
  }));
  const stats = [
    { k: "社交边总数（有向）", v: p.total_edges.toLocaleString() },
    { k: "噪声边（占数量）", v: `${p.noise_edges.toLocaleString()}（${p.noise_count_ratio}%）` },
    { k: "接收者内相对压缩", v: p.relative_compression, note: "<1 表示同一用户的噪声边权重占比被压到低于其数量占比" },
    { k: "噪声边剪枝率（z<0.1）", v: `${p.noise_pruned_ratio}%`, note: `真实边仅 ${p.real_pruned_ratio}% 被剪枝` },
    { k: "噪声边平均权重", v: p.noise_mean_w.toFixed(4) },
    { k: "真实边平均权重", v: p.real_mean_w.toFixed(4) },
    { k: "噪声边平均门值 z", v: p.noise_mean_z.toFixed(3) },
    { k: "真实边平均门值 z", v: p.real_mean_z.toFixed(3) },
  ];
  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-[1fr_1.2fr]">
      <div className="panel p-4">
        <div className="panel-title mb-1">SODRA GRAPH DENOISING · 第四章</div>
        <p className="mb-3 text-xs text-muted-foreground">
          Hard Concrete 可微边门 + 行为锚定注意力 + 双向 Transformer 语义验证，对 10% 注入噪声边的抑制效果（接收者内相对口径）
        </p>
        <div className="grid grid-cols-2 gap-px border border-border bg-border">
          {stats.map((s) => (
            <div key={s.k} className="bg-card p-3" title={s.note}>
              <div className="font-mono text-[10px] text-muted-foreground">{s.k}</div>
              <div className="mt-0.5 font-mono text-lg font-semibold">{s.v}</div>
            </div>
          ))}
        </div>
      </div>
      <div className="panel p-4">
        <div className="panel-title mb-3">边权重分布直方图 · 噪声 vs 真实</div>
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={bins} margin={{ left: -10, right: 8, top: 4 }}>
            <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
            <XAxis dataKey="range" tick={{ fontSize: 9, fontFamily: "JetBrains Mono" }} interval={2} />
            <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} />
            <Tooltip contentStyle={{ fontSize: 11, fontFamily: "JetBrains Mono", background: "hsl(50 44% 94%)", border: "1px solid hsl(70 16% 74%)" }} />
            <Legend wrapperStyle={{ fontSize: 11 }} />
            <Bar dataKey="真实边" fill="#2f6b3a" />
            <Bar dataKey="噪声边" fill="#b98045" />
          </BarChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
}

/* ---------------- 知识图谱浏览 ---------------- */
function KgBrowser() {
  const [q, setQ] = useState("");
  const [selected, setSelected] = useState<number | null>(null);
  const searchQ = trpc.kg.search.useQuery({ q }, { enabled: q.length > 0 });
  const nbQ = trpc.kg.neighbors.useQuery({ entityId: selected! }, { enabled: selected != null });
  const statsQ = trpc.kg.stats.useQuery();

  return (
    <div className="grid grid-cols-1 gap-3 lg:grid-cols-[280px_1fr]">
      <div className="space-y-3">
        <div className="panel p-3">
          <div className="panel-title mb-2">ENTITY SEARCH</div>
          <div className="relative">
            <Search className="absolute left-2 top-2 h-3.5 w-3.5 text-muted-foreground" />
            <Input
              value={q}
              onChange={(e) => setQ(e.target.value)}
              placeholder="搜索实体，如：柑橘"
              className="h-8 rounded-none border-border bg-background pl-7 text-xs"
            />
          </div>
          <div className="mt-1 divide-y divide-border/60">
            {(searchQ.data ?? []).map((e) => (
              <button
                key={e.entityId}
                onClick={() => setSelected(e.entityId)}
                className={`flex w-full items-center justify-between px-1 py-1.5 text-left text-xs hover:bg-secondary ${
                  selected === e.entityId ? "bg-secondary font-medium" : ""
                }`}
              >
                <span>{e.name}</span>
                <span className="font-mono text-[10px] text-muted-foreground">
                  {KG_TYPE_LABEL[e.etype] ?? e.etype}
                </span>
              </button>
            ))}
            {q && searchQ.data?.length === 0 && (
              <div className="px-1 py-2 text-xs text-muted-foreground">无匹配实体</div>
            )}
          </div>
        </div>
        <div className="panel p-3">
          <div className="panel-title mb-2">KG STATS</div>
          <div className="space-y-1 font-mono text-[11px]">
            {(statsQ.data?.byType ?? []).map((t) => (
              <div key={t.etype} className="flex justify-between">
                <span className="text-muted-foreground">{KG_TYPE_LABEL[t.etype] ?? t.etype}</span>
                <span>{t.c}</span>
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="panel p-4">
        <div className="panel-title mb-2">
          {selected != null ? `NEIGHBORHOOD · 实体 #${selected} 的一跳三元组` : "选择左侧实体查看知识关联"}
        </div>
        {selected != null && (
          <div className="max-h-[480px] overflow-y-auto">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-card">
                <tr className="border-b border-border font-mono text-[10px] text-muted-foreground">
                  <th className="py-1 text-left font-normal">HEAD</th>
                  <th className="text-left font-normal">RELATION</th>
                  <th className="text-left font-normal">TAIL</th>
                </tr>
              </thead>
              <tbody>
                {(nbQ.data ?? []).map((t, i) => (
                  <tr key={i} className="border-b border-border/50">
                    <td className="py-1.5">
                      <button className="hover:underline" onClick={() => setSelected(t.h.entityId)}>
                        {t.h.name}
                      </button>
                      <span className="ml-1 font-mono text-[9px] text-muted-foreground">
                        {KG_TYPE_LABEL[t.h.etype] ?? t.h.etype}
                      </span>
                    </td>
                    <td className="font-mono text-[10px] text-[#b98045]">
                      {KG_REL_LABEL[t.r] ?? t.r}
                    </td>
                    <td className="py-1.5">
                      <button className="hover:underline" onClick={() => setSelected(t.t.entityId)}>
                        {t.t.name}
                      </button>
                      <span className="ml-1 font-mono text-[9px] text-muted-foreground">
                        {KG_TYPE_LABEL[t.t.etype] ?? t.t.etype}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            {nbQ.data?.length === 0 && (
              <div className="py-6 text-center text-xs text-muted-foreground">该实体暂无三元组</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

/* ---------------- 场景调度配置 ---------------- */
function SceneConfig() {
  const q = trpc.stats.sceneConfig.useQuery();
  if (q.isLoading) return <Skeleton className="h-64" />;
  const { weights, algo } = (q.data ?? {}) as any;
  return (
    <div className="panel p-4">
      <div className="panel-title mb-3">场景识别规则 → 融合权重（论文 6.3.2 / 6.3.3）</div>
      <table className="w-full text-xs">
        <thead>
          <tr className="border-b border-border font-mono text-[10px] text-muted-foreground">
            <th className="py-1.5 text-left font-normal">场景</th>
            <th className="text-left font-normal">识别规则</th>
            <th className="text-left font-normal">主导算法</th>
            {MODELS.map((m) => (
              <th key={m} className="text-right font-normal">{MODEL_META[m].label} 权重</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {Object.entries(SCENE_META).map(([k, meta]) => (
            <tr key={k} className="border-b border-border/50">
              <td className="py-2 font-medium" style={{ color: meta.color }}>{meta.label}</td>
              <td className="max-w-[280px] text-muted-foreground">{meta.desc}</td>
              <td className="font-mono text-[11px]">{algo?.[k]}</td>
              {MODELS.map((m, i) => (
                <td key={m} className="text-right font-mono">
                  {weights?.[k]?.[i]?.toFixed(2) ?? "—"}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
      <div className="mt-4 grid grid-cols-1 gap-3 md:grid-cols-3">
        {MODELS.map((m) => (
          <div key={m} className="border border-border p-3">
            <div className="font-mono text-[11px] font-semibold" style={{ color: MODEL_META[m].color }}>
              {MODEL_META[m].label}
            </div>
            <div className="mt-0.5 text-xs">{MODEL_META[m].name}</div>
            <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">{MODEL_META[m].thesis}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

export default function Lab() {
  return (
    <div>
      <div className="mb-3">
        <h1 className="text-lg font-semibold">算法工作台</h1>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          论文三、四、五章算法的真实训练结果 · PyTorch 2.8 CPU · 64 维嵌入
        </p>
      </div>
      <Tabs defaultValue="curves">
        <TabsList className="rounded-none border border-border bg-card">
          <TabsTrigger value="curves" className="rounded-none text-xs">训练曲线</TabsTrigger>
          <TabsTrigger value="metrics" className="rounded-none text-xs">指标对比</TabsTrigger>
          <TabsTrigger value="purify" className="rounded-none text-xs">SoDRA 图净化</TabsTrigger>
          <TabsTrigger value="kg" className="rounded-none text-xs">知识图谱</TabsTrigger>
          <TabsTrigger value="scene" className="rounded-none text-xs">场景调度</TabsTrigger>
        </TabsList>
        <TabsContent value="curves" className="mt-3"><Curves /></TabsContent>
        <TabsContent value="metrics" className="mt-3"><Metrics /></TabsContent>
        <TabsContent value="purify" className="mt-3"><Purify /></TabsContent>
        <TabsContent value="kg" className="mt-3"><KgBrowser /></TabsContent>
        <TabsContent value="scene" className="mt-3"><SceneConfig /></TabsContent>
      </Tabs>
    </div>
  );
}
