<script lang="ts">
  import { createEventDispatcher } from 'svelte';
  import FullScreenModal from '../full-screen-modal.svelte';
  import Button from './../../elements/buttons/button.svelte';
  import type { Color } from '$lib/components/elements/buttons/button.svelte';
  import QrCode from 'svelte-qrcode';

  export let link = '';
  export let confirmText = 'Confirm';
  export let confirmColor: Color = 'red';
  export let disabled = false;

  const dispatch = createEventDispatcher<{ cancel: void; confirm: void; 'click-outside': void }>();

  let isConfirmButtonDisabled = false;

  const handleCancel = () => dispatch('cancel');
  const handleEscape = () => {
    if (!isConfirmButtonDisabled) {
      dispatch('cancel');
    }
  };

  const handleConfirm = () => {
    isConfirmButtonDisabled = true;
    dispatch('confirm');
  };

  const handleClickOutside = () => {
    dispatch('click-outside');
  };
</script>

<FullScreenModal on:clickOutside={handleClickOutside} on:escape={() => handleEscape()}>
  <div
    class="w-[500px] max-w-[95vw] rounded-3xl border bg-immich-bg p-4 py-8 shadow-sm dark:border-immich-dark-gray dark:bg-immich-dark-gray dark:text-immich-dark-fg"
  >
    <div>
      <div class="text-md px-4 py-5 text-center">
        <div class="flex justify-center items-center">
          <QrCode value={link} />
        </div>
      </div>

      <div class="mt-4 flex w-full gap-4 px-4">
        <Button color={confirmColor} fullwidth on:click={handleConfirm} disabled={disabled || isConfirmButtonDisabled}>
          {confirmText}
        </Button>
      </div>
    </div>
  </div>
</FullScreenModal>