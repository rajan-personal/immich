import 'package:collection/collection.dart';
import 'package:easy_localization/easy_localization.dart';
import 'package:flutter/material.dart';
import 'package:flutter_hooks/flutter_hooks.dart';
import 'package:hooks_riverpod/hooks_riverpod.dart';
import 'package:immich_mobile/extensions/build_context_extensions.dart';
import 'package:immich_mobile/modules/album/providers/album.provider.dart';
import 'package:immich_mobile/modules/album/ui/album_thumbnail_card.dart';
import 'package:immich_mobile/routing/router.dart';
import 'package:immich_mobile/shared/models/album.dart';
import 'package:immich_mobile/modules/settings/providers/app_settings.provider.dart';
import 'package:immich_mobile/modules/settings/services/app_settings.service.dart';
import 'package:immich_mobile/shared/providers/server_info.provider.dart';
import 'package:immich_mobile/shared/ui/immich_app_bar.dart';

class LibraryPage extends HookConsumerWidget {
  const LibraryPage({Key? key}) : super(key: key);

  @override
  Widget build(BuildContext context, WidgetRef ref) {
    final trashEnabled =
        ref.watch(serverInfoProvider.select((v) => v.serverFeatures.trash));
    final albums = ref.watch(albumProvider);
    var settings = ref.watch(appSettingsServiceProvider);

    useEffect(
      () {
        ref.read(albumProvider.notifier).getAllAlbums();
        return null;
      },
      [],
    );

    final selectedAlbumSortOrder =
        useState(settings.getSetting(AppSettingsEnum.selectedAlbumSortOrder));

    List<Album> sortedAlbums() {
      // Created.
      if (selectedAlbumSortOrder.value == 0) {
        return albums
            .where((a) => a.isRemote)
            .sortedBy((album) => album.createdAt)
            .reversed
            .toList();
      }
      // Album title.
      if (selectedAlbumSortOrder.value == 1) {
        return albums.where((a) => a.isRemote).sortedBy((album) => album.name);
      }
      // Most recent photo, if unset (e.g. empty album, use modifiedAt / updatedAt).
      if (selectedAlbumSortOrder.value == 2) {
        return albums
            .where((a) => a.isRemote)
            .sorted(
              (a, b) => a.lastModifiedAssetTimestamp != null &&
                      b.lastModifiedAssetTimestamp != null
                  ? a.lastModifiedAssetTimestamp!
                      .compareTo(b.lastModifiedAssetTimestamp!)
                  : a.modifiedAt.compareTo(b.modifiedAt),
            )
            .reversed
            .toList();
      }
      // Last modified.
      if (selectedAlbumSortOrder.value == 3) {
        return albums
            .where((a) => a.isRemote)
            .sortedBy((album) => album.modifiedAt)
            .reversed
            .toList();
      }

      // Fallback: Album title.
      return albums.where((a) => a.isRemote).sortedBy((album) => album.name);
    }

    Widget buildSortButton() {
      final options = [
        "library_page_sort_created".tr(),
        "library_page_sort_title".tr(),
        "library_page_sort_most_recent_photo".tr(),
        "library_page_sort_last_modified".tr(),
      ];

      return PopupMenuButton(
        position: PopupMenuPosition.over,
        itemBuilder: (BuildContext context) {
          return options.mapIndexed<PopupMenuEntry<int>>((index, option) {
            final selected = selectedAlbumSortOrder.value == index;
            return PopupMenuItem(
              value: index,
              child: Row(
                children: [
                  Padding(
                    padding: const EdgeInsets.only(right: 12.0),
                    child: Icon(
                      Icons.check,
                      color:
                          selected ? context.primaryColor : Colors.transparent,
                    ),
                  ),
                  Text(
                    option,
                    style: TextStyle(
                      color: selected ? context.primaryColor : null,
                      fontSize: 12.0,
                    ),
                  ),
                ],
              ),
            );
          }).toList();
        },
        onSelected: (int value) {
          selectedAlbumSortOrder.value = value;
          settings.setSetting(AppSettingsEnum.selectedAlbumSortOrder, value);
        },
        child: Row(
          children: [
            Icon(
              Icons.swap_vert_rounded,
              size: 18,
              color: context.primaryColor,
            ),
            Text(
              options[selectedAlbumSortOrder.value],
              style: TextStyle(
                fontWeight: FontWeight.bold,
                color: context.primaryColor,
                fontSize: 12.0,
              ),
            ),
          ],
        ),
      );
    }

    Widget buildCreateAlbumButton() {
      return LayoutBuilder(
        builder: (context, constraints) {
          var cardSize = constraints.maxWidth;

          return GestureDetector(
            onTap: () {
              (context).autoPush(CreateAlbumRoute(isSharedAlbum: false));
            },
            child: Padding(
              padding: const EdgeInsets.only(bottom: 32),
              child: Column(
                mainAxisAlignment: MainAxisAlignment.start,
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  SizedBox(
                    height: cardSize,
                    width: cardSize,
                    child: Card(
                      shape: RoundedRectangleBorder(
                        borderRadius: BorderRadius.circular(20.0),
                      ),
                      child: const Center(
                        child: Icon(
                          Icons.add_rounded,
                          size: 28,
                        ),
                      ),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.only(
                      top: 8.0,
                      bottom: 16,
                      left: 8.0,
                    ),
                    child: const Text(
                      'library_page_new_album',
                      style: TextStyle(
                        fontWeight: FontWeight.bold,
                      ),
                    ).tr(),
                  ),
                ],
              ),
            ),
          );
        },
      );
    }

    Widget buildLibraryNavButton(
      String label,
      IconData icon,
      Function() onClick,
    ) {
      return Expanded(
        child: ElevatedButton.icon(
          onPressed: onClick,
          label: Padding(
            padding: const EdgeInsets.only(left: 8.0),
            child: Text(
              label,
            ),
          ),
          style: context.themeData.elevatedButtonTheme.style?.copyWith(
            alignment: Alignment.centerLeft,
          ),
          icon: Icon(
            icon,
          ),
        ),
      );
    }

    final sorted = sortedAlbums();

    final local = albums.where((a) => a.isLocal).toList();

    Widget? shareTrashButton() {
      return trashEnabled
          ? InkWell(
              onTap: () => context.autoPush(const TrashRoute()),
              borderRadius: BorderRadius.circular(12),
              child: const Icon(
                Icons.delete_rounded,
                size: 25,
              ),
            )
          : null;
    }

    return Scaffold(
      appBar: ImmichAppBar(
        action: shareTrashButton(),
      ),
      body: CustomScrollView(
        slivers: [
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.only(
                left: 12.0,
                right: 12.0,
                top: 24.0,
                bottom: 12.0,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                children: [
                  buildLibraryNavButton(
                      "library_page_favorites".tr(), Icons.favorite_border, () {
                    context.autoNavigate(const FavoritesRoute());
                  }),
                  const SizedBox(width: 12.0),
                  buildLibraryNavButton(
                      "library_page_archive".tr(), Icons.archive_outlined, () {
                    context.autoNavigate(const ArchiveRoute());
                  }),
                ],
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.only(
                top: 12.0,
                left: 12.0,
                right: 12.0,
                bottom: 20.0,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'library_page_albums',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ).tr(),
                  buildSortButton(),
                ],
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(12.0),
            sliver: SliverGrid(
              gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 250,
                crossAxisSpacing: 12,
                childAspectRatio: .7,
              ),
              delegate: SliverChildBuilderDelegate(
                childCount: sorted.length + 1,
                (context, index) {
                  if (index == 0) {
                    return buildCreateAlbumButton();
                  }

                  return AlbumThumbnailCard(
                    album: sorted[index - 1],
                    onTap: () => context.autoPush(
                      AlbumViewerRoute(
                        albumId: sorted[index - 1].id,
                      ),
                    ),
                  );
                },
              ),
            ),
          ),
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.only(
                top: 12.0,
                left: 12.0,
                right: 12.0,
                bottom: 20.0,
              ),
              child: Row(
                mainAxisAlignment: MainAxisAlignment.spaceBetween,
                children: [
                  const Text(
                    'library_page_device_albums',
                    style: TextStyle(fontWeight: FontWeight.bold),
                  ).tr(),
                ],
              ),
            ),
          ),
          SliverPadding(
            padding: const EdgeInsets.all(12.0),
            sliver: SliverGrid(
              gridDelegate: const SliverGridDelegateWithMaxCrossAxisExtent(
                maxCrossAxisExtent: 250,
                mainAxisSpacing: 12,
                crossAxisSpacing: 12,
                childAspectRatio: .7,
              ),
              delegate: SliverChildBuilderDelegate(
                childCount: local.length,
                (context, index) => AlbumThumbnailCard(
                  album: local[index],
                  onTap: () => context.autoPush(
                    AlbumViewerRoute(
                      albumId: local[index].id,
                    ),
                  ),
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }
}
