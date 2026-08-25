import {
    Route,
    Routes,
} from "react-router";

import {
    AppShell,
} from "./components/AppShell";

import {
    HomePage,
} from "./pages/HomePage";

import {
    PredictPage,
} from "./pages/PredictPage";

import {
    ComparePage,
} from "./pages/ComparePage";

import {
    InterpretPage,
} from "./pages/InterpretPage";

import {
    ErrorLabPage,
} from "./pages/ErrorLabPage";

import {
    MetricsPage,
} from "./pages/MetricsPage";

import {
    AboutPage,
} from "./pages/AboutPage";

import {
    NotFoundPage,
} from "./pages/NotFoundPage";

export default function App() {
    return (
        <AppShell>
            <Routes>
                <Route
                    path="/"
                    element={<HomePage />}
                />

                <Route
                    path="/predict"
                    element={<PredictPage />}
                />

                <Route
                    path="/compare"
                    element={<ComparePage />}
                />

                <Route
                    path="/interpret"
                    element={<InterpretPage />}
                />

                <Route
                    path="/errors"
                    element={<ErrorLabPage />}
                />

                <Route
                    path="/metrics"
                    element={<MetricsPage />}
                />

                <Route
                    path="/about"
                    element={<AboutPage />}
                />

                <Route
                    path="*"
                    element={<NotFoundPage />}
                />
            </Routes>
        </AppShell>
    );
}