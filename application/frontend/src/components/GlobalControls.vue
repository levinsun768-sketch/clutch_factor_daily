<script setup lang="ts">
import { inject } from 'vue';

defineProps<{ date: string; universe: string; benchmark: string; locale: string }>();
defineEmits<{
  'update:date': [value: string];
  'update:universe': [value: string];
  'update:benchmark': [value: string];
  'update:locale': [value: string];
}>();
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const universes = ['all', 'hs300', 'csi500', 'csi1000', 'csi2000'];
const benchmarks = ['hs300', 'csi500', 'csi1000'];
</script>

<template>
  <header class="topbar">
    <div class="topbar-title">
      <span>Control Deck</span>
      <strong>{{ universe.toUpperCase() }} / {{ benchmark.toUpperCase() }}</strong>
    </div>
    <div class="topbar-controls">
      <label>
        {{ t('controls.date') }}
        <input class="control" type="date" :value="date" @input="$emit('update:date', ($event.target as HTMLInputElement).value)" />
      </label>
      <label>
        {{ t('controls.universe') }}
        <select class="control" :value="universe" @change="$emit('update:universe', ($event.target as HTMLSelectElement).value)">
          <option v-for="item in universes" :key="item" :value="item">{{ item.toUpperCase() }}</option>
        </select>
      </label>
      <label>
        {{ t('controls.benchmark') }}
        <select class="control" :value="benchmark" @change="$emit('update:benchmark', ($event.target as HTMLSelectElement).value)">
          <option v-for="item in benchmarks" :key="item" :value="item">{{ item.toUpperCase() }}</option>
        </select>
      </label>
      <label>
        {{ t('controls.language') }}
        <select class="control language-select" :value="locale" @change="$emit('update:locale', ($event.target as HTMLSelectElement).value)">
          <option value="zh">{{ t('controls.chinese') }}</option>
          <option value="en">{{ t('controls.english') }}</option>
        </select>
      </label>
    </div>
  </header>
</template>
