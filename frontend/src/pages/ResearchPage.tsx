import { Link } from "react-router";
import {
  HeldoutSection,
  RareFigure,
  SourceLink,
  SummaryTable,
  SeedTable,
  ValidationPlot,
} from "../components/ResearchFigures";
import { TidalDotField } from "../components/TidalDotField";
import { research } from "../data/research";
const CONTENTS = [
  ["abstract", "Abstract"],
  ["question", "Research question"],
  ["data", "Data & preparation"],
  ["methods", "Methods"],
  ["results", "Validation results"],
  ["rare-labels", "Rare emotions"],
  ["heldout", "Held-out test"],
  ["discussion", "Discussion & limits"],
  ["conclusion", "Conclusion"],
  ["evidence", "Evidence & reproducibility"],
];
export function ResearchPage() {
  return (
    <article className="page paper-page">
      <TidalDotField analyzing={false} />
      <header className="paper-heading">
        <p className="eyebrow">
          TmotionLab Research <span className="paper-divider">/</span> Study P11
        </p>
        <h1>
          More examples.
          <br />A better reading?
        </h1>
        <p className="paper-subtitle">
          Rare-Emotion Recognition in Multi-label Text Classification: Real
          Oversampling and Synthetic Augmentation.
        </p>
        <div className="paper-meta">
          <span>Completed exploratory study</span>
          <span>Report v001</span>
          <span>4 strategies · 3 seeds · 28 labels</span>
        </div>
        <div className="inline-links">
          <Link to="/research/experiments">Explore the experiments</Link>
          <SourceLink name="goemotions_research_final_v001.md">
            Download the source report
          </SourceLink>
        </div>
      </header>
      <div className="paper-layout">
        <aside className="paper-index">
          <p className="caption">In this study</p>
          <nav aria-label="Research contents">
            {CONTENTS.map(([id, name], i) => (
              <a key={id} href={`#${id}`}>
                <span>{String(i + 1).padStart(2, "0")}</span>
                {name}
              </a>
            ))}
          </nav>
          <p className="index-note">
            An exploratory project report.
            <br />
            Not a peer-reviewed publication.
          </p>
        </aside>
        <div className="paper-body">
          <section id="abstract" className="research-section abstract">
            <p className="eyebrow">01 / Abstract</p>
            <h2>
              More exposure helped.
              <br />
              <span>Generation was not a clear winner.</span>
            </h2>
            <p>
              Can targeted synthetic examples improve recognition of rare
              emotions beyond simply showing a model more human-annotated
              examples? We compared uniform sampling, real-data oversampling,
              synthetic augmentation and a hybrid schedule using a pinned
              RoBERTa-base model on GoEmotions.
            </p>
            <p>
              Across three paired training seeds, all interventions improved
              validation macro F1 over uniform sampling. Hybrid had the highest
              mean (0.5200), narrowly ahead of Real 4× (0.5173). That difference
              does not establish reliable superiority. Under the predefined
              pragmatic decision rule, the simpler real-data approach was
              selected; its frozen seed-42 model achieved held-out macro F1 of
              0.5237.
            </p>
            <p className="takeaway">
              The useful result is targeted exposure. The evidence for synthetic
              data’s added value is narrower, and strongest descriptively for
              nervousness.
            </p>
            <SourceLink name="multiseed_robustness_v001.json">
              Validation evidence
            </SourceLink>
          </section>
          <section id="question" className="research-section">
            <p className="eyebrow">02 / Introduction</p>
            <h2>What does a rare emotion need?</h2>
            <p>
              A classifier can learn frequent emotions while overlooking labels
              with few positive examples. Multi-label classification makes the
              imbalance harder to interpret: an emotion may coexist with other
              labels, and the learning signal depends on which combinations
              appear during training.
            </p>
            <p>
              This study asks whether targeted synthetic text supplies useful
              information beyond increased exposure to the existing rare-label
              examples. The hypotheses are comparative: targeted sampling should
              improve rare-label recognition relative to uniform draws;
              synthetic augmentation might add variety; a hybrid might combine
              that variety with human-annotated data. The experiment tests those
              expectations through matched training budgets and paired seeds,
              rather than treating additional generated text as an automatic
              improvement.
            </p>
          </section>
          <section id="data" className="research-section">
            <p className="eyebrow">03 / Data &amp; preparation</p>
            <h2>A shared foundation.</h2>
            <p>
              GoEmotions simplified supplies 27 emotion labels plus neutral. An
              example can have multiple labels. From 43,410 official training
              rows, 158 were excluded because their normalized text overlapped
              held-out data, leaving 43,252 training rows. Normalization used
              casefolding and whitespace collapse. Official validation (5,426)
              and test (5,427) memberships were preserved; 11 normalized texts
              overlap between those two splits, an evaluation caveat.
            </p>
            <p>
              The synthetic pilot used GPT-6.1 Terra to generate 300 candidates
              targeting grief, nervousness, pride and relief. One confirmed
              near-duplicate was excluded, leaving 299. Individual human
              semantic annotation was waived as a one-time research decision.
              Generator-proposed secondary labels were not independently
              verified; passing a structural audit does not make them
              human-verified labels.
            </p>
            <div className="inline-links">
              <SourceLink name="data/processed/goemotions_v001/manifest.json">
                Real-data manifest
              </SourceLink>
              <SourceLink name="annotation_audit_v001.json">
                Split-overlap audit
              </SourceLink>
              <SourceLink name="synthetic_pilot_v002_audit.json">
                Synthetic audit
              </SourceLink>
              <SourceLink name="synthetic_emotion_generation_v002.txt">
                Generation prompt
              </SourceLink>
            </div>
            <figure className="research-figure flow-figure">
              <div className="study-flow">
                <div>
                  <span>01 / Prepare</span>
                  <strong>43,252 real examples</strong>
                  <small>+ 299 synthetic candidates</small>
                </div>
                <span aria-hidden="true">→</span>
                <div>
                  <span>02 / Compare</span>
                  <strong>4 strategies × 3 seeds</strong>
                  <small>Shared training budget</small>
                </div>
                <span aria-hidden="true">→</span>
                <div>
                  <span>03 / Freeze</span>
                  <strong>Real 4× · seed 42</strong>
                  <small>Then one held-out test</small>
                </div>
              </div>
              <figcaption>
                Figure 1. Study flow. Validation selected checkpoints and
                thresholds. Test labels were excluded from training, strategy
                selection and threshold tuning.{" "}
                <SourceLink name="final_model_freeze_v001.json">
                  Freeze record
                </SourceLink>
              </figcaption>
            </figure>
          </section>
          <section id="methods" className="research-section">
            <p className="eyebrow">04 / Methods</p>
            <h2>
              Change the exposure.
              <br />
              Hold the budget steady.
            </h2>
            <p>
              All twelve runs fine-tuned RoBERTa-base from a pinned source
              revision. Each run used four epochs, 43,252 draws per epoch and
              5,408 optimizer steps. Seeds 42, 123 and 456 were paired across
              strategies. Deterministic sampling schedules were audited to
              verify the intended draws and epoch iteration.
            </p>
            <figure className="sampling-figure">
              <dl className="method-list">
                <dt>Uniform</dt>
                <dd>
                  Human-annotated real examples sampled uniformly with
                  replacement.
                </dd>
                <dt>Real 4×</dt>
                <dd>
                  Increased sampling weight for real examples containing
                  targeted rare emotions.
                </dd>
                <dt>Synthetic</dt>
                <dd>
                  Targeted draws from the 299 generated examples mixed with real
                  data.
                </dd>
                <dt>Hybrid</dt>
                <dd>
                  A 50/50 mixture of the Real 4× and synthetic sampling
                  schedules.
                </dd>
              </dl>
              <figcaption>
                Figure 2. Sampling strategies. Each strategy draws 43,252
                examples per epoch; the intervention changes exposure, not the
                total training budget.{" "}
                <SourceLink name="controlled_sampling_audit_v001.json">
                  Controlled audit
                </SourceLink>{" "}
                ·{" "}
                <SourceLink name="hybrid_sampling_audit_v001.json">
                  Hybrid audit
                </SourceLink>
              </figcaption>
            </figure>
            <p>
              Checkpoint selection maximized validation macro average precision.
              A global classification threshold was selected per run from a
              fixed validation grid by macro F1. Macro F1 gives each label equal
              weight; micro F1 pools decisions across labels; macro AP measures
              ranking quality without a single classification threshold. Scores
              reported below are validation results, with mean and sample
              standard deviation over three seeds. They are descriptive
              statistics, not significance tests.
            </p>
            <div className="inline-links">
              <SourceLink name="roberta_goemotions_multilabel_real_v001.json">
                Base configuration
              </SourceLink>
              <SourceLink name="multiseed_sampling_audit_v001.json">
                Sampling audit
              </SourceLink>
              <SourceLink name="hybrid_sampling_audit_v001.json">
                Hybrid schedule
              </SourceLink>
            </div>
          </section>
          <section id="results" className="research-section">
            <p className="eyebrow">05 / Validation results</p>
            <h2>The change is in the rare labels.</h2>
            <p>
              All three interventions increased mean macro F1 relative to
              uniform sampling. Micro F1 remained close across strategies,
              illustrating how a pooled metric can conceal changes in less
              frequent labels. Hybrid’s mean macro F1 advantage over Real 4× was
              about 0.0027. With only three seeds and variation at this scale,
              the study does not establish a reliable overall advantage for
              hybrid.
            </p>
            <ValidationPlot />
            <p className="figure-note">
              Figure 3. Validation macro F1 across all twelve runs.
              Full-precision values and per-run thresholds are available in the{" "}
              <Link to="/research/experiments">experiment explorer</Link>.
            </p>
            <SeedTable />
            <SummaryTable />
            <p className="figure-note">
              Table 1. Recomputed from the twelve original metric files; SD uses
              n − 1.{" "}
              <SourceLink name="multiseed_robustness_v001.json">
                Robustness source
              </SourceLink>
            </p>
          </section>
          <section id="rare-labels" className="research-section">
            <p className="eyebrow">06 / Rare emotions</p>
            <h2>A more specific picture.</h2>
            <p>
              Uniform sampling produced zero F1 for all four targeted labels at
              the selected thresholds. This does not mean the model had no
              ranking signal: rare-label AP was nonzero. Targeted exposure made
              positive detections more useful, with results varying by label and
              seed.
            </p>
            <RareFigure />
            <p className="figure-note">
              Figure 4. Rare-label mean validation F1 and positive support.
            </p>
            <p>
              Hybrid improved nervousness F1 relative to Real 4× in all three
              seeds. Other labels were mixed; neither the pattern nor its
              repetition is a significance test. The explorer exposes each
              seed’s paired difference, along with F1, AP and support, so that
              the aggregate does not hide the uncertainty.
            </p>
            <Link to="/research/experiments#rare-labels">
              Inspect rare-label differences
            </Link>
          </section>
          <HeldoutSection />
          <section id="discussion" className="research-section">
            <p className="eyebrow">08 / Discussion &amp; limitations</p>
            <h2>Useful signal, bounded claims.</h2>
            <p>
              Real-data oversampling was selected under the pragmatic rule:
              prefer the simpler human-annotated approach unless hybrid
              demonstrates a meaningful benefit. This is a deployment decision
              under limited evidence, not a proof that synthetic augmentation
              cannot help. Hybrid remains a research alternative. Independent
              labels allow neutral alongside another emotion, or no detections
              at the frozen threshold. Neither case measures a person’s
              emotional state.
            </p>
            <ol className="limitations">
              <li>
                <strong>Three paired seeds.</strong> Small differences cannot
                establish robust superiority or statistical significance.
              </li>
              <li>
                <strong>Very small rare-label supports.</strong> A few decisions
                can substantially change F1. Test grief, for example, has only
                six positives.
              </li>
              <li>
                <strong>Unverified synthetic semantics.</strong> Generated
                secondary labels may introduce noise or systematic bias.
              </li>
              <li>
                <strong>Repeated validation use.</strong> Interventions,
                checkpoints and thresholds were developed on the same validation
                split.
              </li>
              <li>
                <strong>Split overlap.</strong> Eleven normalized texts are
                shared by official validation and test, whose memberships were
                preserved.
              </li>
              <li>
                <strong>Limited test comparison.</strong> The held-out
                evaluation covers only the selected Real 4× model, not all
                strategies.
              </li>
              <li>
                <strong>Transfer to real language.</strong> Short English
                benchmark inputs do not establish reliability across cultures,
                domains, languages or personal circumstances. Sigmoid scores are
                not calibrated confidence or diagnoses.
              </li>
            </ol>
            <p>
              Future work should increase the number of seeds, independently
              review synthetic labels, evaluate out-of-domain text and assess
              calibration on a separate development set. These are proposed
              studies, not completed results. Token attribution would require
              its own validated method; the artistic signal is not evidence of a
              causal model explanation.
            </p>
          </section>
          <section id="conclusion" className="research-section">
            <p className="eyebrow">09 / Conclusion</p>
            <h2>Exposure before complexity.</h2>
            <p>
              Targeted exposure consistently improved validation macro F1.
              Synthetic text provided useful signal for some rare emotions,
              particularly nervousness, without demonstrating a reliable overall
              advantage over real-data oversampling. The frozen Real 4× seed-42
              model is the basis of the live reading studio. Its output remains
              one model’s perspective on language, to be interpreted with human
              context.
            </p>
            <Link to="/">Try a reading</Link>
          </section>
          <section id="evidence" className="research-section" tabIndex={-1}>
            <p className="eyebrow">10 / Evidence &amp; reproducibility</p>
            <h2>Follow the result to its source.</h2>
            <p>
              This edition is generated from saved P11 artifacts. It remains
              readable independently of the inference service. Downloads below
              are exact source copies; SHA-256 digests identify their bytes.
              Model weights and individual held-out predictions are not
              distributed here.
            </p>
            <details>
              <summary>
                Source files &amp; SHA-256 checksums{" "}
                <span className="muted">{research.sources.length} files</span>
              </summary>
              <ol className="source-list">
                {research.sources.map((source) => (
                  <li key={source.path}>
                    <SourceLink name={source.path} />
                    <code>{source.path}</code>
                    <code>SHA-256 {source.sha256}</code>
                  </li>
                ))}
              </ol>
            </details>
            <p>
              All twelve metric files and their experiment provenance are also
              linked from the{" "}
              <Link to="/research/experiments">experiment explorer</Link>.
              Training used <code>{research.runs[0].provenance.model_id}</code>,
              revision <code>{research.runs[0].provenance.model_revision}</code>
              .
            </p>
          </section>
        </div>
      </div>
    </article>
  );
}
