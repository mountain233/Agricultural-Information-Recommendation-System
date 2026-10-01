import {
  mysqlTable,
  serial,
  int,
  varchar,
  text,
  longtext,
  double,
  timestamp,
  bigint,
} from "drizzle-orm/mysql-core";

/** 用户表：对应论文 6.4.1 user 表 */
export const users = mysqlTable("user", {
  id: serial("id").primaryKey(),
  userId: int("user_id").notNull().unique(),          // AgriRec-Sim 中的用户编号
  userType: varchar("user_type", { length: 32 }).notNull(),   // 农户/农技人员/采购商/消费者/合作社管理员
  province: varchar("province", { length: 16 }).notNull(),
  cropType: varchar("crop_type", { length: 16 }).notNull(),   // 关注作物
  farmScale: int("farm_scale").notNull().default(1),
  interactionCount: int("interaction_count").notNull().default(0),
  socialEdgeCount: int("social_edge_count").notNull().default(0),
  noiseEdgeCount: int("noise_edge_count").notNull().default(0),
  isColdStart: int("is_cold_start").notNull().default(0),
  scene: varchar("scene", { length: 32 }).notNull().default("normal"),  // 场景标签
  createdAt: timestamp("created_at").notNull().defaultNow(),
});

/** 物品表：农产品/农资，对应 item 表 */
export const items = mysqlTable("item", {
  id: serial("id").primaryKey(),
  itemId: int("item_id").notNull().unique(),
  name: varchar("name", { length: 128 }).notNull(),
  category: varchar("category", { length: 16 }).notNull(),    // 水果/蔬菜/禽肉蛋品/粮油/农资
  price: double("price").notNull().default(0),
  origin: varchar("origin", { length: 16 }).notNull(),
  crop: varchar("crop", { length: 16 }).notNull(),
  popularity: int("popularity").notNull().default(0),
  isLongTail: int("is_long_tail").notNull().default(0),
  kgEntityId: int("kg_entity_id"),                             // 关联知识图谱实体
});

/** 交互记录（抽样入库，供用户行为时间线展示） */
export const interactions = mysqlTable("interaction", {
  id: serial("id").primaryKey(),
  userId: int("user_id").notNull(),
  itemId: int("item_id").notNull(),
  behaviorType: varchar("behavior_type", { length: 16 }).notNull(), // click/collect/purchase/rating/skip
  rating: int("rating").notNull().default(0),
  ts: bigint("ts", { mode: "number" }).notNull(),
});

/** 社交边（含 SoDRA 净化后的边权重） */
export const socialEdges = mysqlTable("social_edge", {
  id: serial("id").primaryKey(),
  src: int("src").notNull(),
  dst: int("dst").notNull(),
  rel: varchar("rel", { length: 24 }).notNull(),      // FRIEND_OF/COOPERATIVE/TECH_GROUP
  isNoise: int("is_noise").notNull().default(0),
  weight: double("weight").notNull().default(0),      // SoDRA 净化后的边权重
  gate: double("gate").notNull().default(0),          // Hard Concrete 门 z
});

/** 知识图谱实体 */
export const kgEntities = mysqlTable("kg_entity", {
  id: serial("id").primaryKey(),
  entityId: int("entity_id").notNull().unique(),
  name: varchar("name", { length: 64 }).notNull(),
  etype: varchar("etype", { length: 24 }).notNull(),  // Crop/Pest/Fertilizer/Pesticide/Technique/Region/Product
});

/** 知识图谱三元组 */
export const kgTriples = mysqlTable("kg_triple", {
  id: serial("id").primaryKey(),
  h: int("h").notNull(),
  r: varchar("r", { length: 32 }).notNull(),
  t: int("t").notNull(),
});

/** 模型嵌入：每个实体一行，含三个模型的 64 维向量（JSON） */
export const embeddings = mysqlTable("embedding", {
  id: serial("id").primaryKey(),
  entityType: varchar("entity_type", { length: 8 }).notNull(),  // user/item
  entityId: int("entity_id").notNull(),
  socognn: text("socognn").notNull(),
  sodra: text("sodra").notNull(),
  pesatnet: text("pesatnet").notNull(),
});

/** 模型评估指标与训练曲线（JSON 文档行） */
export const modelStats = mysqlTable("model_stats", {
  id: serial("id").primaryKey(),
  key: varchar("key", { length: 64 }).notNull().unique(),   // metrics/curves/purify/dataset_meta/...
  payload: longtext("payload").notNull(),
  updatedAt: timestamp("updated_at").notNull().defaultNow(),
});

/** 用户反馈（在线收集，对应 6.1.2 用户反馈收集模块） */
export const feedback = mysqlTable("feedback", {
  id: serial("id").primaryKey(),
  userId: int("user_id").notNull(),
  itemId: int("item_id").notNull(),
  action: varchar("action", { length: 16 }).notNull(),       // click/collect/purchase/skip/like
  scene: varchar("scene", { length: 32 }).notNull().default("normal"),
  algorithm: varchar("algorithm", { length: 32 }).notNull().default("Fusion"),
  createdAt: timestamp("created_at").notNull().defaultNow(),
});
