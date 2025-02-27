import { AppRoute } from '$lib/constants';
import { api } from '@api';
import { redirect } from '@sveltejs/kit';
import type { PageLoad } from './$types';

export const load = (async ({ url }) => {
  const { data } = await api.serverInfoApi.getServerConfig();
  let joinToken = localStorage.getItem('joinToken');
  if (!data.isInitialized) {
    // Admin not registered
    throw redirect(302, AppRoute.AUTH_REGISTER);
  }

  return {
    meta: {
      title: 'Login',
    },
    joinToken,
  };
}) satisfies PageLoad;
