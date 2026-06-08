<script setup lang="ts">
import { computed, inject } from 'vue';

const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;

const props = defineProps<{
  styles?: Record<string, number | null | undefined>;
  unit?: string;
}>();

const rows = computed(() => {
  const entries = Object.entries(props.styles || {})
    .map(([key, value]) => ({ key, value: Number(value) }))
    .filter((row) => Number.isFinite(row.value));
  const maxAbs = Math.max(1, ...entries.map((row) => Math.abs(row.value)));
  return entries.map((row) => {
    const translated = t(`styleFactors.${row.key}`);
    return {
      ...row,
      displayName: translated === `styleFactors.${row.key}` ? row.key : translated,
      width: `${Math.max(2, (Math.abs(row.value) / maxAbs) * 50)}%`,
      side: row.value >= 0 ? 'positive' : 'negative',
      label: `${row.value >= 0 ? '+' : ''}${row.value.toFixed(1)}${props.unit ? ` ${props.unit}` : ''}`
    };
  });
});
</script>

<template>
  <div class="style-bars-zero">
    <div v-for="row in rows" :key="row.key" class="style-zero-row">
      <span class="style-name" :title="row.key">{{ row.displayName }}</span>
      <div class="zero-track">
        <div class="zero-line"></div>
        <div v-if="row.side === 'negative'" class="bar negative" :style="{ width: row.width }"></div>
        <div v-else class="bar positive" :style="{ width: row.width }"></div>
      </div>
      <b :class="row.side">{{ row.label }}</b>
    </div>
  </div>
</template>
