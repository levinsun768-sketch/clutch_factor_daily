<script setup lang="ts">
import { inject, ref, watch } from 'vue';
import { Send } from '@lucide/vue';

const props = defineProps<{ context: Record<string, unknown> }>();
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const messages = ref([{ role: 'assistant', markdown: t('agent.intro') }]);
const input = ref('');
watch(() => props.context.locale, () => {
  if (messages.value.length === 1 && messages.value[0].role === 'assistant') {
    messages.value[0].markdown = t('agent.intro');
  }
});

async function send() {
  const text = input.value.trim();
  if (!text) return;
  messages.value.push({ role: 'user', markdown: text });
  input.value = '';
  const res = await fetch('/api/agent/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ message: text, context: props.context })
  });
  messages.value.push(await res.json());
}
</script>

<template>
  <aside class="agent-panel">
    <div class="agent-header">
      <div><span class="eyebrow">Artifact Agent</span><strong>{{ t('agent.title') }}</strong></div>
      <i class="live-dot"></i>
    </div>
    <div class="agent-context">
      <span>{{ String(context.route || '/') }}</span>
      <b>{{ String(context.universe || 'all').toUpperCase() }}</b>
    </div>
    <div class="agent-messages">
      <div v-for="(msg, idx) in messages" :key="idx" :class="['agent-message', msg.role === 'user' ? 'user' : 'assistant']">
        {{ msg.markdown }}
      </div>
    </div>
    <div class="agent-input">
      <input v-model="input" :placeholder="t('agent.placeholder')" @keydown.enter="send" />
      <button class="icon-button" @click="send" :title="t('agent.send')"><Send :size="16" /></button>
    </div>
  </aside>
</template>
