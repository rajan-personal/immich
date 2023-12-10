import { AssetEntity, AssetFaceEntity, SmartInfoEntity } from '@app/infra/entities';

export const ISmartInfoRepository = 'ISmartInfoRepository';

export type Embedding = number[];

export interface EmbeddingSearch {
  ownerId: string;
  embedding: Embedding;
  numResults: number;
}

export interface FaceEmbeddingSearch extends EmbeddingSearch {
  maxDistance?: number;
  hasPerson?: boolean;
}

export interface ISmartInfoRepository {
  init(modelName: string): Promise<void>;
  searchCLIP(search: EmbeddingSearch): Promise<AssetEntity[]>;
  searchFaces(search: FaceEmbeddingSearch): Promise<AssetFaceEntity[]>;
  upsert(smartInfo: Partial<SmartInfoEntity>, embedding?: Embedding): Promise<void>;
}
