// Editorial visual families only. Never use this mapping to change API predictions.
export const emotionFamilies = {
  uplift: { name: "Uplift", labels: ["amusement", "excitement", "joy", "optimism", "pride", "relief"] },
  connection: { name: "Connection", labels: ["admiration", "caring", "gratitude", "love"] },
  appraisal: { name: "Appraisal", labels: ["approval", "disapproval"] },
  longing: { name: "Longing", labels: ["desire"] },
  reflection: { name: "Loss & reflection", labels: ["disappointment", "grief", "remorse", "sadness"] },
  alarm: { name: "Alarm & exposure", labels: ["embarrassment", "fear", "nervousness"] },
  friction: { name: "Friction", labels: ["anger", "annoyance", "disgust"] },
  orientation: { name: "Inquiry & orientation", labels: ["confusion", "curiosity", "realization", "surprise"] },
  neutral: { name: "Neutral", labels: ["neutral"] },
} as const;

type EmotionFamily = keyof typeof emotionFamilies;
const labelFamilies = new Map<string, EmotionFamily>(
  Object.entries(emotionFamilies).flatMap(([family, { labels }]) =>
    labels.map(label => [label, family as EmotionFamily] as const)),
);
export const emotionFamily = (label: string): EmotionFamily => labelFamilies.get(label) ?? "neutral";
