import type { ProviderDay, UsageHistory, UsageProvider } from '@waku/client'

/**
 * The per-provider lane order of every `byProvider` array on the wire.
 *
 * The daemon indexes those arrays with `UsageProvider::index`, so this order
 * is a protocol contract: reordering it silently reads one provider's numbers
 * as another's. It must stay in lockstep with the Rust `UsageProvider::ALL`.
 */
export const USAGE_PROVIDERS: readonly UsageProvider[] = [
  'claude',
  'codex',
  'deepSeek',
  'kimi',
  'ohMyPi',
  'pi',
]

/**
 * Display names matching the desktop usage page's `UsageProvider::label`, so
 * both clients name the same lane the same way.
 */
export const USAGE_PROVIDER_LABELS: Record<UsageProvider, string> = {
  claude: 'Claude Code',
  codex: 'Codex',
  deepSeek: 'DeepSeek',
  kimi: 'Kimi Code',
  ohMyPi: 'Oh My Pi',
  pi: 'Pi',
}

/** The provider's chart hue, mirroring the desktop page's `usage_chart_color`. */
export const USAGE_PROVIDER_COLORS: Record<UsageProvider, string> = {
  claude: 'var(--provider-claude)',
  codex: 'var(--provider-codex)',
  deepSeek: 'var(--provider-deepseek)',
  kimi: 'var(--provider-kimi)',
  ohMyPi: 'var(--provider-ohmypi)',
  pi: 'var(--provider-pi)',
}

/**
 * Each provider's lane in a `byProvider` array, derived from `USAGE_PROVIDERS`
 * so the order above stays the single source of truth.
 */
const USAGE_PROVIDER_LANES = Object.fromEntries(
  USAGE_PROVIDERS.map((provider, lane) => [provider, lane]),
) as Record<UsageProvider, number>

const EMPTY_PROVIDER_DAY: ProviderDay = { costUsd: 0, totalTokens: 0 }

/**
 * One provider's amounts out of a `byProvider` lane array. A daemon whose
 * `UsageProvider` is shorter than this client's writes fewer lanes, and those
 * read as zero rather than throwing.
 */
export function usageProviderValue(
  byProvider: readonly ProviderDay[],
  provider: UsageProvider,
  byCost: boolean,
): number {
  const entry = byProvider[USAGE_PROVIDER_LANES[provider]] ?? EMPTY_PROVIDER_DAY
  return byCost ? entry.costUsd : entry.totalTokens
}

/** Whether a row's lane array carries anything for this provider. */
export function providerHasUsage(byProvider: readonly ProviderDay[], provider: UsageProvider): boolean {
  const entry = byProvider[USAGE_PROVIDER_LANES[provider]] ?? EMPTY_PROVIDER_DAY
  return entry.costUsd > 0 || entry.totalTokens > 0
}

/**
 * The providers the window actually used, heaviest first — the ones the chart,
 * its legend, and the per-provider columns draw. Providers with no usage would
 * only add flat zero lines, empty legend chips, and empty columns.
 */
export function usageChartedProviders(history: UsageHistory): UsageProvider[] {
  return history.providers.map((slice) => slice.provider)
}
