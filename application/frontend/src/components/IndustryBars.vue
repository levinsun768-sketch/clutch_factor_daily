<script setup lang="ts">
import { computed } from 'vue';

const props = defineProps<{
  industries?: Array<{ industry: string; premium_bps: number | null; n?: number }>;
}>();

const rows = computed(() => {
  const items = (props.industries || [])
    .map((item) => ({ ...item, value: Number(item.premium_bps), n: Number(item.n || 0) }))
    .filter((item) => Number.isFinite(item.value))
    .sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  const maxAbs = Math.max(1, ...items.map((item) => Math.abs(item.value)));
  const maxN = Math.max(1, ...items.map((item) => item.n));
  return items.map((item, index) => {
    const strength = Math.min(1, Math.abs(item.value) / maxAbs);
    const area = Math.max(0.72, Math.sqrt(item.n / maxN));
    const side = item.value >= 0 ? 'positive' : 'negative';
    return {
      ...item,
      side,
      rank: index + 1,
      label: `${item.value >= 0 ? '+' : ''}${item.value.toFixed(1)} bps`,
      tone: `${0.44 + strength * 0.56}`,
      minHeight: `${76 + area * 74}px`,
      flexGrow: Math.max(0.9, 0.9 + item.n / maxN * 2.8),
    };
  });
});
</script>

<template>
  <div class="industry-heatmap">
    <div
      v-for="row in rows"
      :key="row.industry"
      :class="['industry-tile', row.side]"
      :style="{ '--tile-tone': row.tone, minHeight: row.minHeight, flexGrow: row.flexGrow }"
      :title="`${row.industry} ${row.label}${row.n ? ` · n=${row.n}` : ''}`"
    >
      <div class="tile-head"><strong>{{ row.industry }}</strong><em>#{{ row.rank }}</em></div>
      <div class="tile-foot"><span>{{ row.label }}</span><small v-if="row.n">n={{ row.n }}</small></div>
    </div>
  </div>
</template>
