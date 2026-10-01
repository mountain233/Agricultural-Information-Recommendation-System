import { trpc } from "@/providers/trpc";
import { SCENE_META, BEHAVIOR_LABEL } from "@/lib/consts";
import { Skeleton } from "@/components/ui/skeleton";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend,
} from "recharts";

const TOOLTIP_STYLE = {
  fontSize: 11,
  fontFamily: "JetBrains Mono",
  background: "hsl(50 44% 94%)",
  border: "1px solid hsl(70 16% 74%)",
};

function Stat({ k, v, sub }: { k: string; v: string | number; sub?: string }) {
  return (
    <div className="bg-card p-3" title={sub}>
      <div className="font-mono text-[10px] uppercase tracking-wider text-muted-foreground">{k}</div>
      <div className="mt-0.5 font-mono text-xl font-semibold">{v}</div>
      {sub && <div className="mt-0.5 text-[10px] text-muted-foreground">{sub}</div>}
    </div>
  );
}

export default function Board() {
  const ov = trpc.stats.overview.useQuery();
  const dist = trpc.stats.distributions.useQuery();
  const fb = trpc.reco.feedbackSummary.useQuery();

  if (ov.isLoading) return <Skeleton className="h-96" />;
  const meta = ov.data?.meta as any;

  const sceneData = (dist.data?.userScenes ?? []).map((s) => ({
    name: SCENE_META[s.k]?.label ?? s.k,
    value: s.c,
    color: SCENE_META[s.k]?.color ?? "#888",
  }));
  const catData = (dist.data?.itemCats ?? []).map((c) => ({ name: c.k, 数量: c.c }));
  const typeData = (dist.data?.userTypes ?? []).map((t) => ({ name: t.k, 数量: t.c }));
  const fbData = (fb.data ?? []).map((f) => ({ name: BEHAVIOR_LABEL[f.action] ?? f.action, 次数: f.c }));

  return (
    <div className="space-y-4">
      <div>
        <h1 className="text-lg font-semibold">数据看板</h1>
        <p className="font-mono text-[10px] uppercase tracking-[0.18em] text-muted-foreground">
          AGRIREC-SIM 模拟数据集 · 对齐论文 6.2 实验设置
        </p>
      </div>

      {/* 数据集总览 */}
      <div className="panel p-4">
        <div className="panel-title mb-3">DATASET OVERVIEW</div>
        <div className="grid grid-cols-2 gap-px border border-border bg-border md:grid-cols-4 lg:grid-cols-6">
          <Stat k="用户" v={meta?.users?.toLocaleString()} sub={`冷启动占 ${meta?.cold_ratio?.toFixed(1)}%`} />
          <Stat k="物品" v={meta?.items?.toLocaleString()} sub={`长尾占 ${meta?.long_tail_ratio?.toFixed(1)}%`} />
          <Stat k="交互" v={meta?.interactions?.toLocaleString()} sub={`密度 ${meta?.density?.toFixed(2)}%`} />
          <Stat k="社交边" v={meta?.social_edges?.toLocaleString()} sub={`噪声 ${meta?.noise_ratio?.toFixed(0)}%`} />
          <Stat k="KG 实体" v={meta?.kg_entities?.toLocaleString()} sub={`${meta?.kg_relations} 种关系`} />
          <Stat k="KG 三元组" v={meta?.kg_triples?.toLocaleString()} />
          <Stat k="训练耗时" v={`${Math.round(ov.data?.trainSeconds ?? 0)}s`} sub="三模型 · CPU" />
          <Stat
            k="冷启动覆盖率"
            v={`${((ov.data?.coldCoverage ?? 0) * 100).toFixed(1)}%`}
            sub="SoCoGNN 对冷启动用户的可推荐率"
          />
          <Stat
            k="离线长尾曝光"
            v={`${((ov.data?.longTailExposure ?? 0) * 100).toFixed(1)}%`}
            sub="纯利用策略下为 0 → 服务层 8+2 槽位解决"
          />
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        {/* 用户场景分布 */}
        <div className="panel p-4">
          <div className="panel-title mb-2">用户场景分布（调度入口）</div>
          <ResponsiveContainer width="100%" height={230}>
            <PieChart>
              <Pie data={sceneData} dataKey="value" nameKey="name" innerRadius={48} outerRadius={80} strokeWidth={1} stroke="hsl(51 38% 91%)">
                {sceneData.map((s, i) => (
                  <Cell key={i} fill={s.color} />
                ))}
              </Pie>
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
            </PieChart>
          </ResponsiveContainer>
        </div>

        {/* 用户类型 */}
        <div className="panel p-4">
          <div className="panel-title mb-2">用户类型分布</div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={typeData} margin={{ left: -18, right: 8, top: 4 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="数量" fill="#2f6b3a" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 物品类别 */}
        <div className="panel p-4">
          <div className="panel-title mb-2">物品类别分布</div>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart data={catData} margin={{ left: -18, right: 8, top: 4 }}>
              <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
              <XAxis dataKey="name" tick={{ fontSize: 10 }} />
              <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} />
              <Tooltip contentStyle={TOOLTIP_STYLE} />
              <Bar dataKey="数量" fill="#4a6b2f" />
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* 在线反馈 */}
        <div className="panel p-4">
          <div className="panel-title mb-2">
            在线反馈收集（feedback 表 · {fbData.reduce((a, b) => a + b.次数, 0)} 条）
          </div>
          {fbData.length === 0 ? (
            <div className="flex h-[230px] items-center justify-center border border-dashed border-border text-xs text-muted-foreground">
              暂无反馈 —— 在推荐大厅对推荐结果点击 / 收藏 / 购买 / 跳过
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={230}>
              <BarChart data={fbData} margin={{ left: -18, right: 8, top: 4 }}>
                <CartesianGrid strokeDasharray="2 4" stroke="hsl(70 16% 80%)" />
                <XAxis dataKey="name" tick={{ fontSize: 10 }} />
                <YAxis tick={{ fontSize: 10, fontFamily: "JetBrains Mono" }} allowDecimals={false} />
                <Tooltip contentStyle={TOOLTIP_STYLE} />
                <Bar dataKey="次数" fill="#b98045" />
              </BarChart>
            </ResponsiveContainer>
          )}
        </div>
      </div>
    </div>
  );
}
