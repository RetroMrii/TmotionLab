import { Navigate, Route, Routes } from "react-router";
import { AppShell } from "./components/AppShell";
import { PredictPage } from "./pages/PredictPage";
import { ComparePage } from "./pages/ComparePage";
import { AboutPage } from "./pages/AboutPage";
import { ResearchPage } from "./pages/ResearchPage";
import { ExperimentsPage } from "./pages/ExperimentsPage";
import { NotFoundPage } from "./pages/NotFoundPage";
export default function App() {
  return (
    <AppShell>
      <Routes>
        <Route path="/" element={<PredictPage />} />
        <Route path="/predict" element={<PredictPage />} />
        <Route path="/compare" element={<ComparePage />} />
        <Route path="/research" element={<ResearchPage />} />
        <Route path="/research/experiments" element={<ExperimentsPage />} />
        <Route path="/about" element={<AboutPage />} />
        <Route
          path="/errors"
          element={<Navigate replace to="/research/experiments#historical" />}
        />
        <Route
          path="/metrics"
          element={<Navigate replace to="/research/experiments#historical" />}
        />
        <Route path="/interpret" element={<NotFoundPage legacy />} />
        <Route path="*" element={<NotFoundPage />} />
      </Routes>
    </AppShell>
  );
}
