import {
  createContext,
  useContext,
  useState,
  type ReactNode,
} from "react";

const KEY = "agrirec_demo_user";

const Ctx = createContext<{
  userId: number;
  setUserId: (id: number) => void;
}>({ userId: 5, setUserId: () => {} });

export function DemoUserProvider({ children }: { children: ReactNode }) {
  const [userId, setUserIdState] = useState<number>(() => {
    const v = Number(localStorage.getItem(KEY));
    return Number.isFinite(v) && v >= 0 ? v : 5;
  });
  const setUserId = (id: number) => {
    localStorage.setItem(KEY, String(id));
    setUserIdState(id);
  };
  return <Ctx.Provider value={{ userId, setUserId }}>{children}</Ctx.Provider>;
}

export const useDemoUser = () => useContext(Ctx);
