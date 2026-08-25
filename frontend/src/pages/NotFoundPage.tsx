import {
    Link,
} from "react-router";


export function NotFoundPage() {
    return (
        <div className="page">
            <section className="panel not-found-panel">
                <div className="eyebrow">
                    404
                </div>

                <h1>
                    Page not found
                </h1>

                <p>
                    The requested TmotionLab view does
                    not exist.
                </p>

                <Link
                    className="primary-button inline-button"
                    to="/"
                >
                    Return home
                </Link>
            </section>
        </div>
    );
}