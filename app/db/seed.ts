import { readFileSync } from "fs";
import { join, dirname } from "path";
import { fileURLToPath } from "url";
import { getDb } from "../api/queries/connection";
import {
  users,
  items,
  interactions,
  socialEdges,
  kgEntities,
  kgTriples,
  embeddings,
  modelStats,
} from "./schema";

const DIR = join(dirname(fileURLToPath(import.meta.url)), "seed-data");
const load = (name: string) =>
  JSON.parse(readFileSync(join(DIR, name + ".json"), "utf-8"));

async function batchInsert<T>(
  table: any,
  rows: T[],
  batch: number,
  label: string,
) {
  const db = getDb();
  for (let i = 0; i < rows.length; i += batch) {
    await db.insert(table).values(rows.slice(i, i + batch) as any);
    if ((i / batch) % 20 === 0)
      console.log(`  ${label}: ${Math.min(i + batch, rows.length)}/${rows.length}`);
  }
  console.log(`  ${label}: done (${rows.length})`);
}

async function seed() {
  const db = getDb();
  console.log("Seeding AgriRec-Sim database...");

  // 清空旧数据（按外键无依赖，直接删）
  await db.delete(embeddings);
  await db.delete(modelStats);
  await db.delete(kgTriples);
  await db.delete(kgEntities);
  await db.delete(socialEdges);
  await db.delete(interactions);
  await db.delete(items);
  await db.delete(users);

  const us = load("users").map((u: any) => ({
    userId: u.user_id,
    userType: u.user_type,
    province: u.province,
    cropType: u.crop_type,
    farmScale: u.farm_scale,
    interactionCount: u.interaction_count,
    socialEdgeCount: u.social_edge_count,
    noiseEdgeCount: u.noise_edge_count,
    isColdStart: u.is_cold_start,
    scene: u.scene,
  }));
  await batchInsert(users, us, 500, "users");

  const its = load("items").map((i: any) => ({
    itemId: i.item_id,
    name: i.name,
    category: i.category,
    price: i.price,
    origin: i.origin,
    crop: i.crop,
    popularity: i.popularity ?? 0,
    isLongTail: i.is_long_tail,
    kgEntityId: i.kg_entity_id ?? null,
  }));
  await batchInsert(items, its, 500, "items");

  const ints = load("interactions").map((x: any) => ({
    userId: x.user_id,
    itemId: x.item_id,
    behaviorType: x.behavior_type,
    rating: x.rating ?? 0,
    ts: x.timestamp,
  }));
  await batchInsert(interactions, ints, 1000, "interactions");

  const ses = load("social_edges").map((e: any) => ({
    src: e.src,
    dst: e.dst,
    rel: e.rel,
    isNoise: e.is_noise,
    weight: e.weight,
    gate: e.gate,
  }));
  await batchInsert(socialEdges, ses, 1000, "social_edges");

  const ke = load("kg_entities").map((e: any) => ({
    entityId: e.entity_id,
    name: e.name,
    etype: e.etype,
  }));
  await batchInsert(kgEntities, ke, 500, "kg_entities");

  const kt = load("kg_triples").map((t: any) => ({
    h: t.h,
    r: t.r,
    t: t.t,
  }));
  await batchInsert(kgTriples, kt, 1000, "kg_triples");

  const embs = load("embeddings").map((e: any) => ({
    entityType: e.entity_type,
    entityId: e.entity_id,
    socognn: JSON.stringify(e.socognn),
    sodra: JSON.stringify(e.sodra),
    pesatnet: JSON.stringify(e.pesatnet),
  }));
  await batchInsert(embeddings, embs, 200, "embeddings");

  const stats = load("model_stats");
  for (const [key, payload] of Object.entries(stats)) {
    await db
      .insert(modelStats)
      .values({ key, payload: JSON.stringify(payload) });
  }
  console.log("  model_stats: done");

  console.log("Seed complete.");
  process.exit(0);
}

seed().catch((e) => {
  console.error(e);
  process.exit(1);
});
