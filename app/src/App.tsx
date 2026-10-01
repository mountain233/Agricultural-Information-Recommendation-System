import { Routes, Route } from "react-router";
import { Toaster } from "@/components/ui/sonner";
import AppShell from "./components/AppShell";
import Recommend from "./pages/Recommend";
import Lab from "./pages/Lab";
import Board from "./pages/Board";
import { DemoUserProvider } from "./hooks/useDemoUser";

export default function App() {
  return (
    <DemoUserProvider>
      <Routes>
        <Route element={<AppShell />}>
          <Route path="/" element={<Recommend />} />
          <Route path="/lab" element={<Lab />} />
          <Route path="/board" element={<Board />} />
        </Route>
      </Routes>
      <Toaster position="bottom-right" />
    </DemoUserProvider>
  );
}
