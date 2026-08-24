import { useState } from 'react'
import {
  MessageCircle,
  Send,
  X,
  ShieldCheck,
  Bot,
} from 'lucide-react'

import { api } from '../lib/api'
import { themeFor } from '../lib/portalTheme'

export default function Chatbot({ portal }) {
  const isPayer = portal === 'payer'

  /* Dark chrome, accented per portal — violet on the hospital
     side, blue on the payer side. Matches Layout. */
  const t = themeFor(portal)

  const surface = isPayer ? '#0A1223' : '#120C20'
  const surfaceAlt = isPayer ? '#0C1728' : '#160F28'

  const [open, setOpen] = useState(false)

  const [messages, setMessages] = useState([
    {
      role: 'assistant',
      content: isPayer
        ? 'Hi! I can help you use the PriorAuth AI payer portal.'
        : 'Hi! I can help you use the PriorAuth AI hospital portal.',
    },
  ])

  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)

  // =====================================================
  // SEND MESSAGE
  // =====================================================

  const sendMessage = async () => {
    const text = input.trim()

    if (!text || loading) {
      return
    }

    // Create the user's message
    const userMessage = {
      role: 'user',
      content: text,
    }

    // Add user message to the UI immediately
    setMessages((current) => [
      ...current,
      userMessage,
    ])

    setInput('')
    setLoading(true)

    try {
      /*
       * IMPORTANT
       *
       * Backend expects:
       *
       * {
       *   "messages": [
       *     {
       *       "role": "user",
       *       "content": "hello"
       *     }
       *   ]
       * }
       *
       * The existing `api` client is used here so that
       * the authentication token is automatically included.
       */

      const conversation = [
        ...messages,
        userMessage,
      ]

      const result = await api.post('/api/chat', {
        messages: conversation,
      })

      console.log('CHAT RESPONSE:', result)

      // Backend returns:
      //
      // {
      //   "answer": "...",
      //   "guardrail": false
      // }

      const answer =
        result?.answer ||
        result?.response ||
        result?.message ||
        result?.data?.answer ||
        result?.data?.response ||
        'I could not generate a response.'

      // Add assistant response
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: answer,
        },
      ])
    } catch (error) {
      console.error('CHATBOT ERROR:', error)

      let errorMessage =
        'Sorry, the assistant is temporarily unavailable.'

      // =================================================
      // 401 - Authentication error
      // =================================================

      if (
        error?.status === 401 ||
        error?.response?.status === 401
      ) {
        errorMessage =
          'Your session has expired. Please sign in again.'
      }

      // =================================================
      // 422 - Invalid request
      // =================================================

      else if (
        error?.status === 422 ||
        error?.response?.status === 422
      ) {
        errorMessage =
          'I could not process that message. Please try again.'
      }

      // =================================================
      // 502 - Groq / AI service error
      // =================================================

      else if (
        error?.status === 502 ||
        error?.response?.status === 502
      ) {
        errorMessage =
          'The AI service is currently unavailable. Please try again shortly.'
      }

      // =================================================
      // 503 - Chatbot not configured
      // =================================================

      else if (
        error?.status === 503 ||
        error?.response?.status === 503
      ) {
        errorMessage =
          'The chatbot is not configured yet. Please contact the administrator.'
      }

      // =================================================
      // 500 - Server error
      // =================================================

      else if (
        error?.status === 500 ||
        error?.response?.status === 500
      ) {
        errorMessage =
          'The assistant encountered a server error. Please try again.'
      }

      // =================================================
      // Network error
      // =================================================

      else if (
        error?.message
          ?.toLowerCase()
          ?.includes('network')
      ) {
        errorMessage =
          'Unable to connect to the AI service. Please check that the backend is running.'
      }

      // =================================================
      // Backend detail message
      // =================================================

      else if (error?.detail) {
        errorMessage = error.detail
      }

      // Add error message to chatbot
      setMessages((current) => [
        ...current,
        {
          role: 'assistant',
          content: errorMessage,
        },
      ])
    } finally {
      setLoading(false)
    }
  }

  // =====================================================
  // ENTER KEY HANDLER
  // =====================================================

  const handleKeyDown = (event) => {
    if (
      event.key === 'Enter' &&
      !event.shiftKey
    ) {
      event.preventDefault()
      sendMessage()
    }
  }

  // =====================================================
  // UI
  // =====================================================

  return (
    <>
      {/* =================================================
          FLOATING CHAT BUTTON
          ================================================= */}

      {!open && (
        <button
          onClick={() => setOpen(true)}
          className="fixed bottom-6 right-6 z-50 flex items-center gap-2 rounded-full px-5 py-3 text-sm font-semibold text-white transition hover:scale-105"
          style={{
            backgroundImage: `linear-gradient(135deg, ${t.accent}, ${t.accentDeep})`,
            boxShadow: `0 18px 40px -14px ${t.accentGlow}`,
          }}
        >
          <MessageCircle size={18} />
          Need help?
        </button>
      )}

      {/* =================================================
          CHAT WINDOW
          ================================================= */}

      {open && (
        <div
          className="fixed bottom-6 right-6 z-50 flex h-[560px] w-[370px] max-w-[calc(100vw-32px)] flex-col overflow-hidden rounded-2xl border"
          style={{
            background: surface,
            borderColor: t.lineStrong,
            boxShadow: '0 30px 70px -20px rgba(0,0,0,0.85)',
          }}
        >

          {/* =================================================
              HEADER
              ================================================= */}

          <div
            className="flex items-center justify-between px-4 py-3 text-white"
            style={{
              backgroundImage: `linear-gradient(135deg, ${t.accent}, ${t.accentDeep})`,
            }}
          >
            <div className="flex items-center gap-3">

              <div className="grid h-9 w-9 place-items-center rounded-full bg-white/20">
                <Bot size={19} />
              </div>

              <div>
                <div className="text-sm font-semibold">
                  PriorAuth AI Help
                </div>

                <div className="text-[10px] opacity-80">
                  Platform assistant
                </div>
              </div>

            </div>

            <button
              onClick={() => setOpen(false)}
              className="rounded-lg p-1.5 hover:bg-white/15"
              aria-label="Close chatbot"
            >
              <X size={17} />
            </button>

          </div>

          {/* =================================================
              GUARDRAIL NOTICE
              ================================================= */}

          <div
            className="flex items-start gap-2 border-b px-4 py-2.5"
            style={{
              background: surfaceAlt,
              borderColor: t.line,
            }}
          >

            <ShieldCheck
              size={15}
              className="mt-0.5 shrink-0"
              style={{ color: t.accent }}
            />

            <p
              className="text-[10px] leading-4"
              style={{ color: t.text3 }}
            >
              I can help with PriorAuth AI features,
              requests, documents, coverage, reviews,
              appeals and account-related questions.
            </p>

          </div>

          {/* =================================================
              MESSAGES
              ================================================= */}

          <div
            className="flex-1 space-y-3 overflow-y-auto p-4"
            style={{ background: surface }}
          >

            {messages.map((message, index) => {

              const isUser =
                message.role === 'user'

              return (
                <div
                  key={index}
                  className={`flex ${
                    isUser
                      ? 'justify-end'
                      : 'justify-start'
                  }`}
                >

                  <div
                    className="max-w-[85%] rounded-2xl border px-3.5 py-2.5 text-[12px] leading-5"
                    style={
                      isUser
                        ? {
                            backgroundImage: `linear-gradient(135deg, ${t.accent}, ${t.accentDeep})`,
                            borderColor: 'transparent',
                            color: '#FFFFFF',
                          }
                        : {
                            background: t.panelRaised,
                            borderColor: t.line,
                            color: t.text2,
                          }
                    }
                  >
                    {message.content}
                  </div>

                </div>
              )
            })}

            {/* =================================================
                LOADING
                ================================================= */}

            {loading && (
              <div className="flex justify-start">
                <div
                  className="rounded-2xl border px-4 py-2.5 text-[12px]"
                  style={{
                    background: t.panelRaised,
                    borderColor: t.line,
                    color: t.text3,
                  }}
                >
                  Thinking…
                </div>
              </div>
            )}

          </div>

          {/* =================================================
              INPUT
              ================================================= */}

          <div
            className="border-t p-3"
            style={{
              background: surfaceAlt,
              borderColor: t.line,
            }}
          >

            <div className="flex items-end gap-2">

              <textarea
                value={input}
                onChange={(event) =>
                  setInput(event.target.value)
                }
                onKeyDown={handleKeyDown}
                placeholder={
                  isPayer
                    ? 'Ask about the payer portal...'
                    : 'Ask about the hospital portal...'
                }
                rows={1}
                maxLength={1000}
                disabled={loading}
                className="portal-dark-input min-h-[42px] flex-1 resize-none rounded-lg border px-3.5 py-3 text-[13px] outline-none transition-colors disabled:opacity-50"
                style={{
                  background: t.panelRaised,
                  borderColor: t.line,
                  color: t.text,
                }}
              />

              <button
                onClick={sendMessage}
                disabled={
                  loading ||
                  !input.trim()
                }
                className="grid h-[42px] w-[42px] shrink-0 place-items-center rounded-lg text-white disabled:opacity-40"
                style={{
                  backgroundImage: `linear-gradient(135deg, ${t.accent}, ${t.accentDeep})`,
                }}
                aria-label="Send message"
              >
                <Send size={16} />
              </button>

            </div>

          </div>

        </div>
      )}
    </>
  )
}