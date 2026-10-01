import { useState } from "react";
import { Link, NavLink, Outlet } from "react-router";
import { Sprout, LayoutGrid, FlaskConical, BarChart3, ChevronRight } from "lucide-react";
import { trpc } from "@/providers/trpc";
import { useDemoUser } from "@/hooks/useDemoUser";
import { SCENE_META } from "@/lib/consts";
import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from "@/components/ui/dialog";
import { Badge } from "@/components/ui/badge";

const NAV = [
  { to: "/", label: "推荐大厅", sub: "RECOMMEND", icon: LayoutGrid },
  { to: "/lab", label: "算法工作台", sub: "ALGORITHM LAB", icon: FlaskConical },
  { to: "/board", label: "数据看板", sub: "DATA BOARD", icon: BarChart3 },
];

function UserSwitcher() {
  const { userId, setUserId } = useDemoUser();
  const [open, setOpen] = useState(false);
  const [sceneFilter, setSceneFilter] = useState<string>("");
  const listQ = trpc.user.list.useQuery(
    { scene: sceneFilter || undefined, limit: 24 },
    { enabled: open },
  );
  const profileQ = trpc.user.profile.useQuery({ userId });

  const u = profileQ.data?.user;
  const scene = u ? SCENE_META[u.scene] : null;

  return (
    <>
      <button
        onClick={() => setOpen(true)}
        className="flex items-center gap-3 border border-border bg-card px-3 py-1.5 text-left transition-colors hover:bg-secondary"
      >
        <span className="status-dot pulse-dot bg-[#2f6b3a]" />
        <div className="leading-tight">
          <div className="text-xs font-medium">
            {u ? `${u.userType} · #${u.userId}` : "选择演示用户"}
          </div>
          <div className="font-mono text-[10px] text-muted-foreground">
            {u ? `${u.province} · ${u.cropType}` : "DEMO USER"}
          </div>
        </div>
        {scene && (
          <Badge
            variant="outline"
            className="rounded-none font-mono text-[10px]"
            style={{ color: scene.color, borderColor: scene.color }}
          >
            {scene.label}
          </Badge>
        )}
        <ChevronRight className="h-3.5 w-3.5 text-muted-foreground" />
      </button>

      <Dialog open={open} onOpenChange={setOpen}>
        <DialogContent className="max-w-2xl rounded-none border-border bg-card">
          <DialogHeader>
            <DialogTitle className="text-sm">切换演示用户</DialogTitle>
          </DialogHeader>
          <div className="flex flex-wrap gap-1.5">
            <button
              onClick={() => setSceneFilter("")}
              className={`border px-2 py-0.5 font-mono text-[11px] ${!sceneFilter ? "border-primary bg-primary text-primary-foreground" : "border-border"}`}
            >
              全部场景
            </button>
            {Object.entries(SCENE_META).map(([k, v]) => (
              <button
                key={k}
                onClick={() => setSceneFilter(k)}
                className={`border px-2 py-0.5 font-mono text-[11px] ${sceneFilter === k ? "border-primary bg-primary text-primary-foreground" : "border-border"}`}
              >
                {v.label}
              </button>
            ))}
          </div>
          <div className="grid max-h-[50vh] grid-cols-2 gap-px overflow-y-auto border border-border bg-border">
            {(listQ.data?.rows ?? []).map((r) => (
              <button
                key={r.userId}
                onClick={() => {
                  setUserId(r.userId);
                  setOpen(false);
                }}
                className={`flex items-center justify-between gap-2 bg-card px-3 py-2 text-left text-xs transition-colors hover:bg-secondary ${r.userId === userId ? "bg-secondary" : ""}`}
              >
                <span>
                  <span className="font-medium">
                    #{r.userId} {r.userType}
                  </span>
                  <span className="ml-2 font-mono text-[10px] text-muted-foreground">
                    {r.province} · {r.cropType} · 交互{r.interactionCount}
                  </span>
                </span>
                <span
                  className="font-mono text-[10px]"
                  style={{ color: SCENE_META[r.scene]?.color }}
                >
                  {SCENE_META[r.scene]?.label}
                </span>
              </button>
            ))}
          </div>
          <p className="font-mono text-[10px] text-muted-foreground">
            共 {listQ.data?.total ?? "…"} 位模拟用户 · 选择仅保存在本浏览器
          </p>
        </DialogContent>
      </Dialog>
    </>
  );
}

export default function AppShell() {
  return (
    <div className="flex min-h-screen">
      {/* 左侧导航 */}
      <aside className="flex w-52 shrink-0 flex-col border-r border-border bg-sidebar-background">
        <Link to="/" className="flex items-center gap-2.5 border-b border-border px-4 py-4">
          <Sprout className="h-5 w-5 text-primary" strokeWidth={1.75} />
          <div className="leading-tight">
            <div className="text-sm font-semibold tracking-wide">AgriRec</div>
            <div className="font-mono text-[9px] uppercase tracking-[0.2em] text-muted-foreground">
              农业信息推荐系统
            </div>
          </div>
        </Link>
        <nav className="flex-1 py-3">
          {NAV.map((n) => (
            <NavLink
              key={n.to}
              to={n.to}
              end={n.to === "/"}
              className={({ isActive }) =>
                `group flex items-center gap-3 border-l-2 px-4 py-2.5 transition-colors ${
                  isActive
                    ? "border-primary bg-sidebar-accent text-foreground"
                    : "border-transparent text-muted-foreground hover:bg-sidebar-accent/60 hover:text-foreground"
                }`
              }
            >
              <n.icon className="h-4 w-4" strokeWidth={1.5} />
              <span className="leading-tight">
                <span className="block text-[13px]">{n.label}</span>
                <span className="block font-mono text-[9px] uppercase tracking-[0.18em] opacity-60">
                  {n.sub}
                </span>
              </span>
            </NavLink>
          ))}
        </nav>
        <div className="border-t border-border px-4 py-3 font-mono text-[9px] leading-relaxed text-muted-foreground">
          <div>AGRIREC-SIM · 10K USERS</div>
          <div>SoCoGNN / SoDRA / PESatNet</div>
          <div className="mt-1 flex items-center gap-1.5">
            <span className="status-dot bg-[#2f6b3a]" />
            云端数据库已连接
          </div>
        </div>
      </aside>

      {/* 主区 */}
      <div className="flex min-w-0 flex-1 flex-col">
        <header className="flex items-center justify-between border-b border-border px-5 py-2.5">
          <div className="font-mono text-[10px] uppercase tracking-[0.22em] text-muted-foreground">
            稀疏与噪声场景的图神经网络推荐 · 农业应用
          </div>
          <UserSwitcher />
        </header>
        <main className="min-w-0 flex-1 p-5">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
