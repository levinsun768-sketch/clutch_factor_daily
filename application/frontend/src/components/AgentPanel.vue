<script setup lang="ts">
import { inject, ref, watch } from 'vue';
import { Send } from '@lucide/vue';

type AgentArtifact = { type: string; title: string; route: string };
type ChatMessage = { role: 'assistant' | 'user'; markdown: string; artifacts?: AgentArtifact[] };
const props = defineProps<{ context: Record<string, unknown> }>();
const i18n = inject<any>('i18n') || { t: (key: string) => key };
const t = i18n.t;
const messages = ref<ChatMessage[]>([{ role: 'assistant', markdown: t('agent.intro') }]);
const loading = ref(false);
const input = ref('');

watch(() => props.context.locale, () => {
  if (messages.value.length === 1 && messages.value[0].role === 'assistant') {
    messages.value[0].markdown = t('agent.intro');
  }
});

async function send() {
  const text = input.value.trim();
  if (!text || loading.value) return;
  messages.value.push({ role: 'user', markdown: text });
  input.value = '';
  loading.value = true;
  try {
    const res = await fetch('/api/agent/chat', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: text, context: props.context })
    });
    const payload = await res.json();
    messages.value.push({
      role: 'assistant',
      markdown: payload.markdown || JSON.stringify(payload, null, 2),
      artifacts: (payload.artifacts || []) as AgentArtifact[],
    });
  } catch (error) {
    messages.value.push({ role: 'assistant', markdown: String(error) });
  } finally {
    loading.value = false;
  }
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
        <div class="agent-markdown">{{ msg.markdown }}</div>
        <div v-if="msg.artifacts?.length" class="agent-artifacts">
          <RouterLink v-for="(artifact, artifactIdx) in msg.artifacts" :key="artifactIdx" :to="artifact.route" class="agent-artifact">{{ artifact.title }}</RouterLink>
        </div>
      </div>
      <div v-if="loading" class="agent-message assistant">{{ t('agent.loading') }}</div>
    </div>
    <div class="agent-input">
      <input v-model="input" :placeholder="t('agent.placeholder')" @keydown.enter="send" />
      <button class="icon-button" @click="send" :title="t('agent.send')"><Send :size="16" /></button>
    </div>
  </aside>
</template>
