/** 场景与算法的展示元数据（对应论文 6.3.2 场景化调度） */
export const SCENE_META: Record<
  string,
  { label: string; algo: string; desc: string; color: string }
> = {
  cold_start: {
    label: "冷启动",
    algo: "SoCoGNN",
    desc: "行为稀疏，以社交同质性补全协同信号（ADVF 双视图融合）",
    color: "#2f6b3a",
  },
  noisy_social: {
    label: "社交噪声",
    algo: "SoDRA",
    desc: "社交关系含噪声，启用可微边门降噪与语义验证（SoDRA）",
    color: "#b98045",
  },
  knowledge_rich: {
    label: "知识丰富",
    algo: "PESatNet",
    desc: "行为充分，知识图谱标量关系编码提供主要增益（PESatNet）",
    color: "#4a6b2f",
  },
  normal: {
    label: "常规",
    algo: "Fusion",
    desc: "三模型等权融合，兼顾鲁棒性与精度",
    color: "#40654a",
  },
};

export type ModelKey = "socognn" | "sodra" | "pesatnet";

export const MODEL_META: Record<
  ModelKey,
  { label: string; name: string; thesis: string; color: string }
> = {
  socognn: {
    label: "SoCoGNN",
    name: "稀疏感知社交协同图神经网络",
    thesis: "第三章 · 面向数据稀疏场景",
    color: "#2f6b3a",
  },
  sodra: {
    label: "SoDRA",
    name: "社交噪声感知双视图鲁棒推荐",
    thesis: "第四章 · 面向社交噪声场景",
    color: "#b98045",
  },
  pesatnet: {
    label: "PESatNet",
    name: "参数高效的知识增强卫星网络",
    thesis: "第五章 · 知识增强与高效推理",
    color: "#4a6b2f",
  },
};

export const BEHAVIOR_LABEL: Record<string, string> = {
  click: "点击",
  collect: "收藏",
  purchase: "购买",
  rating: "评分",
  skip: "跳过",
};

export const KG_TYPE_LABEL: Record<string, string> = {
  Crop: "作物",
  Pest: "病虫害",
  Fertilizer: "肥料",
  Pesticide: "农药",
  Technique: "农技",
  Region: "产区",
  Product: "产品",
};

export const KG_REL_LABEL: Record<string, string> = {
  SUSCEPTIBLE_TO: "易感染",
  CONTROLLED_BY: "可用农药防治",
  TREATED_BY: "可用技术防治",
  NEEDS_FERTILIZER: "需要肥料",
  SUITABLE_REGION: "适宜产区",
  GROWS_IN: "种植于",
  RELATED_CROP: "关联作物",
  BELONGS_TO: "属于",
  COMPANION: "伴生",
  CONFLICTS_WITH: "相克",
  SEASON_OF: "产季相关",
  RELATED_TECHNIQUE: "关联农技",
};
