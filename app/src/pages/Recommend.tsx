import { useState } from "react";
import { trpc } from "@/providers/trpc";
import { useDemoUser } from "@/hooks/useDemoUser";
import { SCENE_META, MODEL_META, BEHAVIOR_LABEL, type ModelKey } from "@/lib/consts";
import { Badge } from "@/components/ui/badge";
import { Skeleton } from "@/components/ui/skeleton";
import { toast } from "sonner";
import { MousePointerClick, Bookmark, ShoppingCart, EyeOff, RefreshCw, Network } from "lucide-react";

type Rec = {
  itemId: number;
  score: number;
  perModel: { socognn: number; sodra: number; pesatnet: number };
  source: string;
  slot: "exploit" | "explore";
  item: {
    name: string;
    category: string;
    price: number;
    origin: string;
    crop: string;
    popularity: number;
    isLongTail: number;
  } | null;
};

function ModelBars({ pm }: { pm: Rec["perModel"] }) {
  return (
    <div className="space-y-1">
      {(Object.keys(MODEL_META) as ModelKey[]).map((m) => (
        <div key={m} className="flex items-center gap-1.5">
          <span className="w-14 shrink-0 font-mono text-[9px] text-muted-foreground">
            {MODEL_META[m].label}
          </span>
          <div className="h-1 flex-1 bg-muted">
            <div
              className="h-full"
              style={{ width: `${Math.round(pm[m] * 100)}%`, background: MODEL_META[m].color }}
            />
          </div>
          <span className="w-8 text-right font-mono text-[9px] text-muted-foreground">
            {(pm[m] * 100).toFixed(0)}
          </span>
        </div>
      ))}
    </div>
  );
}

function RecCard({ rec, userId, scene }: { rec: Rec; userId: number; scene: string }) {
  const utils = trpc.useUtils();
  const fb = trpc.reco.submitFeedback.useMutation({
    onSuccess: (_d, v) => {
      toast.success(`已记录「${BEHAVIOR_LABEL[v.action]}」反馈`, {
        description: `物品 #${v.itemId} · 写入云端 feedback 表`,
      });
      utils.reco.feedbackSummary.invalidate();
    },
  });
  const [acted, setActed] = useState<string | null>(null);
  if (!rec.item) return null;
  const it = rec.item;
  const act = (action: "click" | "collect" | "purchase" | "skip") => {
    setActed(action);
    fb.mutate({ userId, itemId: rec.itemId, action, scene, algorithm: rec.source });
  };
  return (
    <div className="panel group flex flex-col p-3 transition-colors hover:bg-secondary/40">
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0">
          <div className="truncate text-[13px] font-medium" title={it.name}>
            {it.name}
          </div>
          <div className="mt-0.5 font-mono text-[10px] text-muted-foreground">
            {it.category} · {it.crop} · 热度 {it.popularity}
          </div>
        </div>
        <div className="text-right">
          <div className="font-mono text-lg font-semibold leading-none">
            {(rec.score * 100).toFixed(1)}
          </div>
          <div className="font-mono text-[9px] uppercase tracking-wider text-muted-foreground">
            匹配分
          </div>
        </div>
      </div>

      <div className="mt-2 flex flex-wrap gap-1">
        <Badge
          variant="outline"
          className="rounded-none border-current font-mono text-[9px]"
          style={{ color: Object.values(MODEL_META).find((m) => m.label === rec.source)?.color }}
        >
          {rec.source}
        </Badge>
        {rec.slot === "explore" && (
          <Badge variant="outline" className="rounded-none border-[#b98045] font-mono text-[9px] text-[#b98045]">
            长尾探索
          </Badge>
        )}
        {it.isLongTail === 1 && rec.slot === "exploit" && (
          <Badge variant="outline" className="rounded-none font-mono text-[9px] text-muted-foreground">
            长尾物品
          </Badge>
        )}
      </div>

      <div className="mt-2.5">
        <ModelBars pm={rec.perModel} />
      </div>

      <div className="mt-auto flex items-center justify-between border-t border-border pt-2" style={{ marginTop: 10 }}>
        <span className="font-mono text-[13px] font-medium">¥{it.price.toFixed(2)}</span>
        <div className="flex gap-0.5">
          {(
            [
              ["click", MousePointerClick, "点击"],
              ["collect", Bookmark, "收藏"],
              ["purchase", ShoppingCart, "购买"],
              ["skip", EyeOff, "跳过"],
            ] as const
          ).map(([a, Icon, label]) => (
            <button
              key={a}
              title={label}
              disabled={fb.isPending}
              onClick={() => act(a)}
              className={`p-1.5 transition-colors hover:bg-primary hover:text-primary-foreground ${
                acted === a ? "bg-primary text-primary-foreground" : "text-muted-foreground"
              }`}
            >
              <Icon className="h-3.5 w-3.5" strokeWidth={1.5} />
            </button>
          ))}
        </div>
      </div>
    </div>
  );
}

export default function Recommend() {
  const { userId } = useDemoUser();
  const q = trpc.reco.recommend.useQuery({ userId, k: 10 });
  const d = q.data;
  const scene = d ? SCENE_META[d.scene] : null;

  return (
    <div className="space-y-4">
      {/* 场景调度条（论文 6.3.2） */}
      <div className="panel grid grid-cols-1 gap-0 md:grid-cols-[1fr_auto]">
        <div className="border-border p-4 md:border-r">
          <div className="panel-title">SCENE DISPATCH · 场景化调度</div>
          {q.isLoading || !d || !scene ? (
            <Skeleton className="mt-2 h-8 w-64" />
          ) : (
            <>
              <div className="mt-1.5 flex items-baseline gap-3">
                <span className="text-xl font-semibold" style={{ color: scene.color }}>
                  {scene.label}场景
                </span>
                <span className="font-mono text-[11px] text-muted-foreground">
                  主导算法 → {d.mainAlgo}
                </span>
              </div>
              <p className="mt-1 text-xs text-muted-foreground">{scene.desc}</p>
              <div className="mt-2 font-mono text-[10px] text-muted-foreground">
                用户 #{d.user.userId} · {d.user.userType} · {d.user.province} · 关注
                {d.user.cropType} · 交互 {d.user.interactionCount} 次 · 社交边{" "}
                {d.user.socialEdgeCount}（噪声 {d.user.noiseEdgeCount}）
              </div>
            </>
          )}
        </div>
        {/* 融合权重（论文 6.3.3） */}
        <div className="w-full p-4 md:w-72">
          <div className="panel-title">FUSION WEIGHTS · 6.3.3 加权融合</div>
          {d && (
            <div className="mt-2 space-y-1.5">
              {(Object.keys(MODEL_META) as ModelKey[]).map((m) => (
                <div key={m} className="flex items-center gap-2">
                  <span className="w-16 font-mono text-[10px]">{MODEL_META[m].label}</span>
                  <div className="h-2 flex-1 bg-muted">
                    <div
                      className="h-full transition-all"
                      style={{
                        width: `${d.weights[m] * 100}%`,
                        background: MODEL_META[m].color,
                      }}
                    />
                  </div>
                  <span className="w-10 text-right font-mono text-[10px]">
                    {d.weights[m].toFixed(2)}
                  </span>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>

      {/* 推荐结果 */}
      <div>
        <div className="mb-2 flex items-center justify-between">
          <div className="panel-title">RECOMMENDATIONS · 8 利用 + 2 长尾探索</div>
          <button
            onClick={() => q.refetch()}
            className="flex items-center gap-1 font-mono text-[10px] text-muted-foreground hover:text-foreground"
          >
            <RefreshCw className="h-3 w-3" /> 刷新
          </button>
        </div>
        {q.isLoading ? (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {Array.from({ length: 10 }).map((_, i) => (
              <Skeleton key={i} className="h-44" />
            ))}
          </div>
        ) : q.isError ? (
          <div className="panel p-6 text-sm text-destructive">加载失败：{q.error.message}</div>
        ) : (
          <div className="grid grid-cols-1 gap-2 sm:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5">
            {d!.recs.map((r) => (
              <RecCard key={r.itemId} rec={r as Rec} userId={userId} scene={d!.scene} />
            ))}
          </div>
        )}
        <p className="mt-2 flex items-center gap-1.5 font-mono text-[10px] text-muted-foreground">
          <Network className="h-3 w-3" />
          三模型嵌入点积 → 各模型 z-score → sigmoid 归一 → 按场景权重加权 →
          过滤历史行为 → 8 条利用 + 2 条长尾探索槽位
        </p>
      </div>
    </div>
  );
}
