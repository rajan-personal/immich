import { getAuthUser } from '$lib/utils/auth';
import { api, ThumbnailFormat } from '@api';
import { error } from '@sveltejs/kit';
import type { AxiosError } from 'axios';
import type { PageLoad } from './$types';
import { AppRoute } from '$lib/constants';
import { goto } from '$app/navigation';

export const load = (async ({ params }) => {
  if (localStorage.getItem('joinToken')) localStorage.removeItem('joinToken')
  const { key } = params;
  const user = await getAuthUser();
  if (!user) {
    localStorage.setItem('joinToken', key);
    goto(AppRoute.AUTH_LOGIN);
  }
  else api.sharedLinkApi.getAlbumAccess({ token: key }).then(res => goto(AppRoute.ALBUMS + '/' + res.data)).catch((err: AxiosError) => console.log(err));

  try {
    const { data: sharedLink } = await api.sharedLinkApi.getMySharedLink({ key });

    const assetCount = sharedLink.assets.length;
    const assetId = sharedLink.album?.albumThumbnailAssetId || sharedLink.assets[0]?.id;

    return {
      sharedLink,
      meta: {
        title: sharedLink.album ? sharedLink.album.albumName : 'Public Share',
        description: sharedLink.description || `${assetCount} shared photos & videos.`,
        imageUrl: assetId
          ? api.getAssetThumbnailUrl(assetId, ThumbnailFormat.Webp, sharedLink.key)
          : '/feature-panel.png',
      },
    };
  } catch (e) {
    // handle unauthorized error
    // TODO this doesn't allow for 404 shared links anymore
    if ((e as AxiosError).response?.status === 401) {
      return {
        passwordRequired: true,
        sharedLinkKey: key,
        meta: {
          title: 'Password Required',
        },
      };
    }

    throw error(404, {
      message: 'Invalid shared link',
    });
  }
}) satisfies PageLoad;
