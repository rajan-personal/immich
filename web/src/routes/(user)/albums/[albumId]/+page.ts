import { authenticate } from '$lib/utils/auth';
import { api } from '@api';
import type { PageLoad } from './$types';

export const load = (async ({ params }) => {
  await authenticate();
  const { data: album } = await api.albumApi.getAlbumInfo({ id: params.albumId, withoutAssets: true });
  const { data: response } = await api.personApi.getAllPeopleFromAlbum({ id: params.albumId, withHidden: false });

  return {
    album,
    meta: {
      title: album.albumName,
    },
    response,
  };
}) satisfies PageLoad;
