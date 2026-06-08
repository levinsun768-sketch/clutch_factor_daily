<script setup lang="ts">
import { computed } from 'vue';

const props = withDefaults(defineProps<{
  points: Array<{ date: string; value: number }>;
  strokeWidth?: number;
  height?: number;
  showValue?: boolean;
  label?: string;
}>(), {
  strokeWidth: 1.25,
  height: 48,
  showValue: false,
  label: '',
});

const cleanPoints = computed(() => props.points.filter((p) => Number.isFinite(Number(p.value))));
const summary = computed(() => {
  const values = cleanPoints.value.map((p) => Number(p.value));
  if (!values.length) return { min: 0, max: 0, last: null as number | null, firstDate: '', lastDate: '' };
  return {
    min: Math.min(...values),
    max: Math.max(...values),
    last: values[values.length - 1],
    firstDate: cleanPoints.value[0]?.date || '',
    lastDate: cleanPoints.value[cleanPoints.value.length - 1]?.date || '',
  };
});
const polyline = computed(() => {
  const values = cleanPoints.value.map((p) => Number(p.value));
  if (values.length < 2) return '';
  const span = Math.max(summary.value.max - summary.value.min, 1e-9);
  return values.map((v, i) => `${(i / (values.length - 1)) * 120},${36 - ((v - summary.value.min) / span) * 32}`).join(' ');
});
const lastLabel = computed(() => {
  const value = summary.value.last;
  if (value === null) return '-';
  return Math.abs(value) < 0.2 ? value.toFixed(4) : value.toFixed(2);
});
</script>

<template>
  <div class="sparkline-wrap" :title="summary.firstDate && summary.lastDate ? `${summary.firstDate} to ${summary.lastDate}` : ''">
    <div v-if="showValue" class="sparkline-meta">
      <span>{{ label }}</span>
      <strong>{{ lastLabel }}</strong>
    </div>
    <svg class="sparkline" :style="{ height: `${height}px` }" viewBox="0 0 120 40" preserveAspectRatio="none">
      <line v-if="summary.min < 0 && summary.max > 0" x1="0" x2="120" :y1="36 - ((0 - summary.min) / Math.max(summary.max - summary.min, 1e-9)) * 32" :y2="36 - ((0 - summary.min) / Math.max(summary.max - summary.min, 1e-9)) * 32" class="sparkline-zero" />
      <polyline :points="polyline" fill="none" stroke="currentColor" :stroke-width="strokeWidth" vector-effect="non-scaling-stroke" />
    </svg>
  </div>
</template>
