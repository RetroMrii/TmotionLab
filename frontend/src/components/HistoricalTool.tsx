import { type ReactNode, useEffect, useState } from "react";
import { getModels } from "../api/client";

export function HistoricalTool({ capability, children }: { capability: string; children: ReactNode }) {
    const [status, setStatus] = useState("loading");
    useEffect(() => {
        let active = true;
        getModels().then(({ models }) => {
            if (active) setStatus(models.some(model => model.dataset === "dair_ai_emotion"
                && model.capabilities.includes(capability)) ? "available" : "unavailable");
        }).catch(() => { if (active) setStatus("unavailable"); });
        return () => { active = false; };
    }, [capability]);
    return <>
        <div className="alert alert-info historical-note">Historical DAIR.AI laboratory · six-label, single-label benchmark.
            These results use a different dataset and task from GoEmotions and are not comparable as the same benchmark.</div>
        {status === "available" ? children : <section className="panel" role="status">
            <h1>{status === "loading" ? "Checking historical tool…" : "Historical tool unavailable"}</h1>
            {status !== "loading" && <p>{capability === "live_interpret"
                ? "The archived TextCNN runtime is unavailable. GoEmotions RoBERTa does not provide TextCNN activation-span interpretation."
                : "The API or required historical artifacts are unavailable. No benchmark results can be shown."}</p>}
        </section>}
    </>;
}
