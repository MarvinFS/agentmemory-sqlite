import Anthropic from '@anthropic-ai/sdk'
import type { MemoryProvider } from '../types.js'
import { getEnvVar } from '../config.js'

type Effort = NonNullable<Anthropic.OutputConfig['effort']>

export class AnthropicProvider implements MemoryProvider {
  name = 'anthropic'
  private client: Anthropic
  private model: string
  private maxTokens: number
  private effort?: Effort

  constructor(apiKey: string, model: string, maxTokens: number, baseURL?: string) {
    this.client = new Anthropic({ apiKey, ...(baseURL ? { baseURL } : {}) })
    this.model = model
    this.maxTokens = maxTokens
    // ANTHROPIC_EFFORT -> output_config.effort, mirroring OPENAI_REASONING_EFFORT
    // in openai.ts. Unset leaves the model at its own default, which on Claude
    // Opus 5 means adaptive thinking is on: measured at 18s on a hard
    // consolidation against the hard 30s compress timeout in consolidate.ts,
    // versus 11s at "medium". Levels are model-specific - Haiku 4.5 rejects the
    // parameter outright and only Opus 5 accepts "xhigh".
    this.effort = (getEnvVar('ANTHROPIC_EFFORT') || undefined) as Effort | undefined
  }

  private get outputConfig() {
    return this.effort ? { output_config: { effort: this.effort } } : {}
  }

  async compress(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt)
  }

  async summarize(systemPrompt: string, userPrompt: string): Promise<string> {
    return this.call(systemPrompt, userPrompt)
  }

  async describeImage(imageData: string, mimeType: string, prompt: string): Promise<string> {
    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: this.maxTokens,
      ...this.outputConfig,
      messages: [{
        role: 'user',
        content: [
          {
            type: 'image',
            source: { type: 'base64', media_type: mimeType as 'image/png' | 'image/jpeg' | 'image/gif' | 'image/webp', data: imageData },
          },
          { type: 'text', text: prompt },
        ],
      }],
    })

    const textBlock = response.content.find((b) => b.type === 'text')
    return textBlock?.text ?? ''
  }

  private async call(systemPrompt: string, userPrompt: string): Promise<string> {
    const response = await this.client.messages.create({
      model: this.model,
      max_tokens: this.maxTokens,
      ...this.outputConfig,
      system: systemPrompt,
      messages: [{ role: 'user', content: userPrompt }],
    })

    const textBlock = response.content.find((b) => b.type === 'text')
    return textBlock?.text ?? ''
  }
}
