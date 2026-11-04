import { useEffect, useState } from "react";
import { getConfusions, getErrors, getMetrics } from "../api/client";
import type {
  ConfusionResponse,
  ErrorListResponse,
  MetricsResponse,
} from "../types/api";
import { TableScroll } from "./ResearchFigures";
export function HistoricalEvidence() {
  const [data, setData] = useState<{
    metrics: MetricsResponse;
    errors: ErrorListResponse;
    confusions: ConfusionResponse;
  } | null>(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    let active = true;
    Promise.all([getMetrics(), getErrors({ limit: 20 }), getConfusions(10)])
      .then(([metrics, errors, confusions]) => {
        if (active) setData({ metrics, errors, confusions });
      })
      .catch(() => {
        if (active) setError(true);
      });
    return () => {
      active = false;
    };
  }, []);
  return (
    <section
      id="historical"
      className="research-section historical-section"
      tabIndex={-1}
    >
      <p className="eyebrow">Earlier chapter / DAIR.AI</p>
      <h2>Historical evidence.</h2>
      <p>
        Six-label single-label benchmark metrics and saved error cases from the
        earlier model family. These values are separate from the GoEmotions
        validation and test results above.
      </p>
      {error ? (
        <p role="status" className="notice">
          Historical evidence is unavailable while the artifact API cannot be
          reached. The P11 study above remains available.
        </p>
      ) : !data ? (
        <p role="status">Loading historical evidence…</p>
      ) : (
        <>
          <TableScroll label="Historical model metrics">
            <table>
              <caption>Historical DAIR.AI model metrics</caption>
              <thead>
                <tr>
                  <th scope="col">Model</th>
                  <th scope="col">Accuracy</th>
                  <th scope="col">Macro F1</th>
                  <th scope="col">Weighted F1</th>
                </tr>
              </thead>
              <tbody>
                {data.metrics.models.map((model) => (
                  <tr key={model.model_id}>
                    <th scope="row">{model.model_id}</th>
                    <td>{model.accuracy.toFixed(4)}</td>
                    <td>{model.macro_f1.toFixed(4)}</td>
                    <td>{model.weighted_f1.toFixed(4)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </TableScroll>
          <details id="historical-errors">
            <summary>
              Saved error cases{" "}
              <span className="muted">
                {data.errors.total} total · first {data.errors.items.length}
              </span>
            </summary>
            <p>{data.errors.benchmark_note}</p>
            <TableScroll label="Historical error cases">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Example</th>
                    <th scope="col">Benchmark</th>
                    <th scope="col">TextCNN prediction</th>
                    <th scope="col">Raw score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.errors.items.map((item) => (
                    <tr key={item.example_id}>
                      <th scope="row">
                        {item.example_id}
                      </th>
                      <td>{item.true_label_name}</td>
                      <td>{item.textcnn_prediction}</td>
                      <td>{item.textcnn_score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableScroll>
          </details>
          <details>
            <summary>Most frequent confusion pairs</summary>
            <p>{data.confusions.benchmark_note}</p>
            <TableScroll label="Historical confusion pairs">
              <table>
                <thead>
                  <tr>
                    <th scope="col">Benchmark → prediction</th>
                    <th scope="col">Count</th>
                    <th scope="col">Mean raw score</th>
                  </tr>
                </thead>
                <tbody>
                  {data.confusions.items.map((item) => (
                    <tr key={item.confusion_pair}>
                      <th scope="row">
                        {item.true_label_name} → {item.textcnn_prediction}
                      </th>
                      <td>{item.count}</td>
                      <td>{item.mean_textcnn_score.toFixed(4)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </TableScroll>
          </details>
        </>
      )}
    </section>
  );
}
