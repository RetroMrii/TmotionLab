import type {
    ReactNode,
} from "react";

import {
    NavLink,
} from "react-router";

interface AppShellProps {
    children: ReactNode;
}

const navigation = [
    {
        path: "/",
        label: "Home",
    },
    {
        path: "/predict",
        label: "Predict",
    },
    {
        path: "/compare",
        label: "Compare",
    },
    {
        path: "/interpret",
        label: "Interpret",
    },
    {
        path: "/errors",
        label: "Error Lab",
    },
    {
        path: "/metrics",
        label: "Metrics",
    },
    {
        path: "/about",
        label: "About",
    },
];

export function AppShell({
    children,
}: AppShellProps) {
    return (
        <div className="app-layout">
            <aside className="sidebar">
                <div className="brand-block">
                    <div className="brand-mark">
                        TM
                    </div>

                    <div>
                        <div className="brand-name">
                            TmotionLab
                        </div>

                        <div className="brand-subtitle">
                            NLP model laboratory
                        </div>
                    </div>
                </div>

                <nav
                    className="navigation"
                    aria-label="Primary navigation"
                >
                    {navigation.map(
                        ({ path, label }) => (
                            <NavLink
                                key={path}
                                to={path}
                                end={path === "/"}
                                className={({
                                    isActive,
                                }) =>
                                    isActive
                                        ? "nav-link nav-link-active"
                                        : "nav-link"
                                }
                            >
                                {label}
                            </NavLink>
                        ),
                    )}
                </nav>

                <div className="sidebar-footer">
                    <span>
                        Educational ML system
                    </span>

                    <span>
                        Not a diagnostic tool
                    </span>
                </div>
            </aside>

            <main
                className="main-content"
                id="main-content"
            >
                {children}
            </main>
        </div>
    );
}