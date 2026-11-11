import { useShallow } from "zustand/react/shallow"

import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select"
import { Separator } from "@/components/ui/separator"
import { Sheet, SheetContent, SheetDescription, SheetHeader, SheetTitle } from "@/components/ui/sheet"
import { Slider } from "@/components/ui/slider"
import { Switch } from "@/components/ui/switch"
import { Textarea } from "@/components/ui/textarea"
import { useChat } from "@/lib/store"
import type { Settings } from "@/lib/types"

export function SettingsSheet({ open, onOpenChange }: { open: boolean; onOpenChange: (open: boolean) => void }) {
  const { settings, setSettings, resetSettings } = useChat(
    useShallow((s) => ({ settings: s.settings, setSettings: s.setSettings, resetSettings: s.resetSettings })),
  )
  const set = <K extends keyof Settings>(key: K) => (value: Settings[K]) => setSettings({ [key]: value } as Partial<Settings>)

  return (
    <Sheet open={open} onOpenChange={onOpenChange}>
      <SheetContent className="w-full gap-0 sm:max-w-sm">
        <SheetHeader className="border-b">
          <SheetTitle>Settings</SheetTitle>
          <SheetDescription className="sr-only">Request parameters for this playground.</SheetDescription>
        </SheetHeader>

        <div className="flex-1 space-y-4 overflow-y-auto px-4 py-4">
          <Field label="System instructions">
            <Textarea
              value={settings.system}
              onChange={(e) => setSettings({ system: e.target.value })}
              placeholder="You are a helpful assistant… (the cat will ignore this)"
              className="min-h-20 resize-y"
            />
          </Field>

          <Separator />

          <SliderField label="Temperature" value={settings.temperature} display={settings.temperature.toFixed(2)} min={0} max={2} step={0.05} onChange={set("temperature")} />
          <div className="grid grid-cols-2 gap-3">
            <Field label="Max tokens">
              <Input type="number" min={1} value={settings.maxTokens} onChange={(e) => setSettings({ maxTokens: e.target.value })} placeholder="Auto" />
            </Field>
            <Field label="Seed">
              <Input type="number" value={settings.seed} onChange={(e) => setSettings({ seed: e.target.value })} placeholder="Random" />
            </Field>
            <Field label="Reasoning">
              <Select value={settings.reasoning} onValueChange={(v) => setSettings({ reasoning: v as Settings["reasoning"] })}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="default">Default</SelectItem>
                  <SelectItem value="none">None</SelectItem>
                  <SelectItem value="low">Low</SelectItem>
                  <SelectItem value="medium">Medium</SelectItem>
                  <SelectItem value="high">High</SelectItem>
                </SelectContent>
              </Select>
            </Field>
            <Field label="Response format">
              <Select value={settings.format} onValueChange={(v) => setSettings({ format: v as Settings["format"] })}>
                <SelectTrigger className="w-full">
                  <SelectValue />
                </SelectTrigger>
                <SelectContent>
                  <SelectItem value="text">Text</SelectItem>
                  <SelectItem value="json_object">JSON</SelectItem>
                </SelectContent>
              </Select>
            </Field>
          </div>
          <SwitchRow id="stream" label="Stream responses" checked={settings.stream} onChange={set("stream")} />

          <Separator />

          <SwitchRow id="tools" label="Use tools" checked={settings.tools} onChange={(on) => setSettings({ tools: on, image: on ? false : settings.image })} />
          <Field label="Tool choice">
            <Select value={settings.toolChoice} onValueChange={(v) => setSettings({ toolChoice: v as Settings["toolChoice"] })}>
              <SelectTrigger className="w-full">
                <SelectValue />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="auto">Auto</SelectItem>
                <SelectItem value="required">Required</SelectItem>
                <SelectItem value="none">None</SelectItem>
              </SelectContent>
            </Select>
          </Field>

          <Separator />

          <SliderField
            label="Time to first token"
            value={settings.ttft}
            display={settings.ttft ? `${settings.ttft} ms` : "instant"}
            min={0}
            max={3000}
            step={50}
            onChange={set("ttft")}
          />
          <SliderField label="Speed" value={settings.tps} display={settings.tps ? `${settings.tps} tok/s` : "instant"} min={0} max={400} step={10} onChange={set("tps")} />

          <Separator />

          <Field label="API key" hint="If the server needs one">
            <Input type="password" autoComplete="off" value={settings.apiKey} onChange={(e) => setSettings({ apiKey: e.target.value })} placeholder="sk-…" />
          </Field>
          <Button variant="ghost" className="w-full text-muted-foreground" onClick={resetSettings}>
            Reset to defaults
          </Button>
        </div>
      </SheetContent>
    </Sheet>
  )
}

function Field({ label, hint, children }: { label: string; hint?: string; children: React.ReactNode }) {
  return (
    <div className="space-y-2">
      <div className="flex items-baseline justify-between gap-2">
        <Label>{label}</Label>
        {hint && <span className="text-xs text-muted-foreground">{hint}</span>}
      </div>
      {children}
    </div>
  )
}

function SliderField({
  label,
  value,
  display,
  min,
  max,
  step,
  onChange,
}: {
  label: string
  value: number
  display: string
  min: number
  max: number
  step: number
  onChange: (v: number) => void
}) {
  return (
    <div className="space-y-3">
      <div className="flex items-baseline justify-between">
        <Label>{label}</Label>
        <span className="font-mono text-xs text-muted-foreground tabular-nums">{display}</span>
      </div>
      <Slider value={[value]} min={min} max={max} step={step} onValueChange={([v]) => onChange(v)} />
    </div>
  )
}

function SwitchRow({ id, label, checked, onChange }: { id: string; label: string; checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <div className="flex items-center justify-between">
      <Label htmlFor={id}>{label}</Label>
      <Switch id={id} checked={checked} onCheckedChange={onChange} />
    </div>
  )
}
