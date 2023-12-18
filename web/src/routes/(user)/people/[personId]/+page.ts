import { authenticate } from '$lib/utils/auth';
import { api } from '@api';
import type { PageLoad } from './$types';

export const load = (async ({ params, url }) => {
  const user = await authenticate();
  const albumId = url.searchParams.get('albumId') ||  undefined;

  const { data: person } = await api.personApi.getPerson({ id: params.personId });
  const { data: statistics } = await api.personApi.getPersonStatistics({ id: params.personId });

  return {
    albumId,
    user,
    person,
    statistics,
    meta: {
      title: person.name || 'Person',
    },
  };
}) satisfies PageLoad;
