import { authenticate } from '$lib/utils/auth';
import { api } from '@api';
import type { PageLoad } from './$types';

export const load = (async () => {
  await authenticate();
  // const { data: albums } = await api.albumApi.getAllAlbums();
  const { data: nsalbums } = await api.albumApi.getAllAlbums({ shared: false });
  const { data: salbums } = await api.albumApi.getAllAlbums({ shared: true });

  const albums = [...nsalbums,...salbums];

  return {
    albums,
    meta: {
      title: 'Albums',
    },
  };
}) satisfies PageLoad;
