import { Link } from "react-router";
export function NotFoundPage({ legacy = false }: { legacy?: boolean }) {
  return (
    <div className="page fallback-page">
      <p className="eyebrow">
        {legacy ? "Historical tool" : "404 / A different path"}
      </p>
      <h1>
        {legacy
          ? "This view has reached its limit."
          : "There’s no reading here."}
      </h1>
      <p className="intro">
        {legacy
          ? "Historical TextCNN interpretation is unavailable. The current GoEmotions model does not provide token attribution."
          : "The page you’re looking for may have moved. The studio is a good place to begin again."}
      </p>
      <div className="inline-links">
        <Link className="primary-button" to="/">
          Back to Analyze
        </Link>
        <Link to={legacy ? "/research/experiments#historical" : "/research"}>
          {legacy ? "Explore historical evidence" : "Explore the research"}
        </Link>
      </div>
    </div>
  );
}
