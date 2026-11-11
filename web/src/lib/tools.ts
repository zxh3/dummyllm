/** Demo tools offered to the model. They "run" in the browser with canned results. */
export const TOOLS = [
  {
    type: "function",
    function: {
      name: "get_weather",
      description: "Get the current weather for a city.",
      parameters: {
        type: "object",
        properties: {
          city: { type: "string" },
          unit: { type: "string", enum: ["celsius", "fahrenheit"] },
        },
        required: ["city", "unit"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "get_time",
      description: "Get the current time in a timezone.",
      parameters: {
        type: "object",
        properties: { timezone: { type: "string", description: "IANA timezone, e.g. Europe/Paris" } },
        required: ["timezone"],
        additionalProperties: false,
      },
    },
  },
  {
    type: "function",
    function: {
      name: "search_web",
      description: "Search the web.",
      parameters: {
        type: "object",
        properties: {
          query: { type: "string" },
          max_results: { type: "integer", minimum: 1, maximum: 10 },
        },
        required: ["query"],
        additionalProperties: false,
      },
    },
  },
] as const

export const TOOL_NAMES = TOOLS.map((t) => t.function.name)

const pick = <T,>(xs: readonly T[]) => xs[Math.floor(Math.random() * xs.length)]
const between = (lo: number, hi: number) => Math.round(lo + Math.random() * (hi - lo))

export function runTool(name: string, argText: string): unknown {
  let args: Record<string, unknown> = {}
  try {
    args = JSON.parse(argText || "{}")
  } catch {
    return { error: "arguments were not valid JSON (the cat sat on the keyboard)" }
  }
  switch (name) {
    case "get_weather":
      return {
        city: args.city,
        temperature: between(-4, 34),
        unit: args.unit ?? "celsius",
        conditions: pick([
          "purrfectly sunny",
          "cloudy with a chance of hairballs",
          "raining cats and dogs",
          "foggy, ideal for napping",
        ]),
      }
    case "get_time":
      return {
        timezone: args.timezone,
        local_time: new Date().toLocaleTimeString("en-US", { hour: "2-digit", minute: "2-digit" }),
        note: "It is always nap time.",
      }
    case "search_web":
      return {
        query: args.query,
        results: [
          { title: "Why do cats knock things off tables?", url: "https://meow.cat/science/gravity" },
          { title: "The definitive ranking of cardboard boxes", url: "https://meow.cat/boxes" },
          { title: "Laser pointers: a philosophical inquiry", url: "https://meow.cat/red-dot" },
        ].slice(0, Number(args.max_results) || 3),
      }
    default:
      return { ok: true }
  }
}
