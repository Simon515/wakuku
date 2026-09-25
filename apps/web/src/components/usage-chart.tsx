import { areaY, crosshair, defineChart, lineY } from '@tanstack/charts'
import { decorative } from '@tanstack/charts/mark/decorative'
import { Chart } from '@tanstack/charts/react'
import { scaleLinear } from '@tanstack/charts/scales/linear'
import { tooltip } from '@tanstack/charts/tooltip'
import type { UsageHistory } from '@waku/client'
import { scaleUtc } from 'd3-scale'
import { useI18n, type AppLocale } from '@/lib/i18n'
import {
  USAGE_PROVIDER_COLORS,
  USAGE_PROVIDER_LABELS,
  usageChartedProviders,
  usageProviderValue,
} from '@/lib/usage-presentation'

export type UsageMetric = 'cost' | 'tokens'

interface UsageChartRow {
  id: string
  date: Date
  provider: string
  value: number
}

export function UsageTrendChart({
  history,
  metric,
}: {
  history: UsageHistory
  metric: UsageMetric
}) {
  const { locale, t } = useI18n()
  // Only the providers this window used get a band: a provider with no usage
  // would only draw a flat zero line and an empty tooltip row.
  const charted = usageChartedProviders(history)
  const rows: UsageChartRow[] = history.daily.flatMap((day) =>
    charted.map((provider) => ({
      id: `${provider}-${day.day}`,
      date: parseUsageDay(day.day),
      provider: USAGE_PROVIDER_LABELS[provider],
      value: usageProviderValue(day.byProvider, provider, metric === 'cost'),
    })),
  )

  const definition = defineChart({
    marks: [
      decorative(areaY(rows, {
        id: 'usage-area',
        x: 'date',
        y: 'value',
        z: 'provider',
        color: 'provider',
        key: 'id',
        fillOpacity: 0.1,
      })),
      lineY(rows, {
        id: 'usage-line',
        x: 'date',
        y: 'value',
        z: 'provider',
        color: 'provider',
        key: 'id',
        strokeWidth: 1.8,
      }),
      crosshair({
        x: { label: false, stroke: 'var(--text-ghost)', strokeOpacity: 0.7 },
        y: false,
      }),
    ],
    x: {
      scale: scaleUtc,
      axis: {
        line: false,
        ticks: {
          count: 3,
          size: 0,
          padding: 8,
          format: (value) => formatChartDay(value, locale),
        },
        tickLabels: { thin: { priority: 'ends', minGap: 24 }, fontSize: 10 },
      },
    },
    y: {
      scale: scaleLinear,
      nice: true,
      grid: true,
      axis: {
        line: false,
        ticks: {
          count: 4,
          size: 0,
          padding: 8,
          format: (value) => metric === 'cost'
            ? formatCompactMoney(value, locale)
            : formatCompactNumber(value, locale),
        },
        tickLabels: { fontSize: 10 },
      },
    },
    color: {
      domain: charted.map((provider) => USAGE_PROVIDER_LABELS[provider]),
      range: charted.map((provider) => USAGE_PROVIDER_COLORS[provider]),
    },
    focus: 'group-x',
    maxFocusDistance: Number.POSITIVE_INFINITY,
    tooltip,
  })

  if (!rows.length) {
    return (
      <div className="grid h-[188px] place-items-center text-[11.5px] text-[var(--text-tertiary)]">
        {t('usage.no_activity_window')}
      </div>
    )
  }

  return (
    <Chart
      ariaDescription={t('usage.chart_description', {
        metric: t(metric === 'cost' ? 'usage.cost' : 'usage.processed_tokens'),
      })}
      ariaLabel={t(metric === 'cost' ? 'usage.daily_cost' : 'usage.daily_processed_tokens')}
      className="waku-usage-chart"
      definition={definition}
      height={188}
      initialWidth={640}
    />
  )
}

function parseUsageDay(day: string) {
  return new Date(`${day}T00:00:00Z`)
}

function formatChartDay(value: Date, locale: AppLocale) {
  return new Intl.DateTimeFormat(locale, { month: 'short', day: 'numeric', timeZone: 'UTC' }).format(value)
}

function formatCompactMoney(value: number, locale: AppLocale) {
  if (!value) return '$0'
  return new Intl.NumberFormat(locale, {
    style: 'currency',
    currency: 'USD',
    notation: value >= 1_000 ? 'compact' : 'standard',
    maximumFractionDigits: value < 10 ? 1 : 0,
  }).format(value)
}

function formatCompactNumber(value: number, locale: AppLocale) {
  if (!value) return '0'
  return new Intl.NumberFormat(locale, { notation: 'compact', maximumFractionDigits: 1 }).format(value)
}
