import type {
  IPrimitivePaneRenderer,
  IPrimitivePaneView,
  ISeriesPrimitive,
  SeriesAttachedParameter,
  Time,
  UTCTimestamp,
} from 'lightweight-charts'
import type { ForecastPoint } from './analysis'

type BandCoordinate = { x: number; upper: number; lower: number }

/** Draw only the region between the forecast bounds, without changing price autoscale. */
export class ForecastBand implements ISeriesPrimitive<Time> {
  private chart: SeriesAttachedParameter<Time>['chart'] | null = null
  private series: SeriesAttachedParameter<Time>['series'] | null = null
  private requestUpdate: (() => void) | null = null
  private points: readonly ForecastPoint[] = []
  private coordinates: BandCoordinate[] = []
  private color = 'rgba(142, 180, 212, 0.12)'
  private visible = true

  private readonly renderer: IPrimitivePaneRenderer = {
    draw: (target) => {
      if (!this.visible || this.coordinates.length < 2) return
      target.useMediaCoordinateSpace(({ context, mediaSize }) => {
        context.save()
        // A wide error band is clipped to the pane, never allowed to flatten the candles.
        context.beginPath()
        context.rect(0, 0, mediaSize.width, mediaSize.height)
        context.clip()
        context.beginPath()
        this.coordinates.forEach((point, index) => {
          if (index === 0) context.moveTo(point.x, point.upper)
          else context.lineTo(point.x, point.upper)
        })
        for (let index = this.coordinates.length - 1; index >= 0; index -= 1) {
          const point = this.coordinates[index]
          context.lineTo(point.x, point.lower)
        }
        context.closePath()
        context.fillStyle = this.color
        context.fill()
        context.restore()
      })
    },
  }

  private readonly view: IPrimitivePaneView = {
    zOrder: () => 'bottom',
    renderer: () => this.renderer,
  }

  attached({ chart, series, requestUpdate }: SeriesAttachedParameter<Time>): void {
    this.chart = chart
    this.series = series
    this.requestUpdate = requestUpdate
    this.requestUpdate()
  }

  detached(): void {
    this.chart = null
    this.series = null
    this.requestUpdate = null
    this.coordinates = []
  }

  setData(points: readonly ForecastPoint[], color: string, visible: boolean): void {
    if (this.points === points && this.color === color && this.visible === visible) return
    this.points = points
    this.color = color
    this.visible = visible
    this.requestUpdate?.()
  }

  updateAllViews(): void {
    const chart = this.chart
    const series = this.series
    if (!chart || !series || !this.visible) {
      this.coordinates = []
      return
    }
    this.coordinates = this.points.flatMap((point) => {
      const x = chart.timeScale().timeToCoordinate(point.time as UTCTimestamp)
      const upper = series.priceToCoordinate(point.upper)
      const lower = series.priceToCoordinate(point.lower)
      if (x === null || upper === null || lower === null) return []
      return [{ x, upper, lower }]
    })
  }

  paneViews(): readonly IPrimitivePaneView[] {
    return [this.view]
  }
}
