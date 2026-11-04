import { Link } from "react-router";
import { LensMark } from "../components/AppShell";
export function AboutPage() {
  return (
    <article className="page about-page">
      <header className="about-heading">
        <p className="eyebrow">About TmotionLab</p>
        <h1>
          Language has layers.
          <br />
          <span>Look a little closer.</span>
        </h1>
        <p className="intro">
          A small space to explore how machine learning reads emotion in
          language—and where that reading ends.
        </p>
        <div className="about-symbol">
          <LensMark />
          <span>
            A model perspective.
            <br />
            Always human context.
          </span>
        </div>
      </header>
      <section className="about-chapter">
        <div className="chapter-rail">
          <span>01</span>
          <p>The idea</p>
        </div>
        <div>
          <h2>Words carry more than one thing.</h2>
          <p>
            A thank-you can hold relief. A goodbye can carry excitement and
            sadness at once. TmotionLab starts with that ambiguity: a passage
            can invite several readings, and no single score contains its whole
            meaning.
          </p>
          <p>
            The reading studio makes a model’s response visible. Write a
            passage, inspect the ranked emotions, then look behind the reading
            at the full scores and the text the model actually received. The
            purpose is curiosity and comparison, with human context close at
            hand.
          </p>
        </div>
      </section>
      <section className="about-chapter">
        <div className="chapter-rail">
          <span>02</span>
          <p>The models</p>
        </div>
        <div>
          <h2>From six labels to twenty-eight.</h2>
          <p>
            The project’s earlier chapter used the six-label DAIR.AI emotion
            benchmark. A TF-IDF baseline, an educational NumPy neural network
            and a TextCNN offered three ways to classify the same saved passage.
            These models use a single-label softmax over six emotions, with no
            neutral class and uncalibrated scores. Their historical predictions
            and benchmark evidence remain available in Compare.
          </p>
          <p>
            The current reading studio uses a RoBERTa model trained on
            GoEmotions: 27 emotions plus neutral, with independent scores so
            labels can coexist. It is the frozen Real 4× seed-42 model selected
            in Study P11. A validation-selected threshold of 0.25 defines
            detections; rank order alone does not.
          </p>
        </div>
      </section>
      <blockquote className="about-quote">
        “A reading is a starting point.
        <br />
        <span>Context is what gives it meaning.</span>”
      </blockquote>
      <section className="about-chapter">
        <div className="chapter-rail">
          <span>03</span>
          <p>The research</p>
        </div>
        <div>
          <h2>A result should be inspectable.</h2>
          <p>
            The research edition follows a practical question: do rare emotions
            benefit more from generated examples or from more exposure to
            human-annotated data? Twelve controlled runs compared four sampling
            strategies across three seeds.
          </p>
          <p>
            The publication explains the study and its limits. The explorer lets
            you move between individual runs, means, variation and rare-label
            comparisons. Macro F1 weights labels equally; micro F1 pools label
            decisions; exact-match accuracy requires the entire predicted label
            set to match. They measure different aspects of performance. Every
            result connects to saved metrics and provenance, independently of
            live API availability. The selected model’s held-out test stays
            separate from the validation comparisons that informed its
            selection.
          </p>
          <div className="inline-links">
            <Link to="/research">Read the study</Link>
            <Link to="/research/experiments">Explore the experiments</Link>
          </div>
        </div>
      </section>
      <section className="about-chapter">
        <div className="chapter-rail">
          <span>04</span>
          <p>The limits</p>
        </div>
        <div>
          <h2>A perspective, with boundaries.</h2>
          <p>
            These are uncalibrated model scores, not measured emotional states
            or diagnoses. A model can misread irony, cultural context, personal
            history or language outside its training distribution. The live
            model reads English text and only the first 48 tokens, including
            special tokens.
          </p>
          <p>
            The optical signal is an artistic response to the ranked scores. It
            is not attention, an explanation of the model, or word-level
            attribution. The current GoEmotions model does not provide token
            attribution. Historical TextCNN interpretation is unavailable.
          </p>
          <details>
            <summary>Technical notes</summary>
            <p>
              Live inference uses independent sigmoid scores across 28 labels.
              The interface shows the returned detection set and the highest
              valid ranked scores without forcing a single emotional answer.
              Empty detection sets and neutral alongside another label are
              possible.
            </p>
            <p>
              Your text is sent to the configured inference service when you
              select Analyze. The interface does not save your passages in
              browser storage. The theme preference is stored locally. This
              describes frontend behavior, not a server retention policy.
            </p>
          </details>
        </div>
      </section>
      <div className="about-close">
        <h2>Start with a few words.</h2>
        <Link className="primary-button" to="/">
          Open the reading studio
        </Link>
      </div>
    </article>
  );
}
