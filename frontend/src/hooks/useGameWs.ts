import { useEffect, useRef, useState, useCallback } from 'react';
import type { GameWsMessage } from '@/types/game';

import SockJS from 'sockjs-client';
import { Client, type IMessage } from '@stomp/stompjs';

const WS_BASE_URL = import.meta.env.VITE_WS_BASE_URL || 'https://localhost:8080/ws';

interface UseGameWsOptions {
  onConnect?: () => void;
  onDisconnect?: () => void;
  onError?: (error: Error) => void;
  onFeedback?: (msg: GameWsMessage) => void;
}

interface UseGameWsReturn {
  isConnected: boolean;
  isConnecting: boolean;
  connect: (sessionId: string) => void;
  disconnect: () => void;
  sendFrame: (params: { sessionId: string; blob: Blob; currentPlayTime: number }) => Promise<void>;
  sendPoseData: (params: { sessionId: string; poseData: number[][]; currentPlayTime: number }) => void;
  clientRef: React.MutableRefObject<Client | null>;
}

/** Blob -> Base64 (prefix 없이 본문만 반환) */
async function blobToBase64(blob: Blob): Promise<string> {
  const buf = await blob.arrayBuffer();
  let binary = '';
  const bytes = new Uint8Array(buf);
  const len = bytes.byteLength;
  for (let i = 0; i < len; i++) binary += String.fromCharCode(bytes[i]);
  return btoa(binary);
}

export function useGameWs(options?: UseGameWsOptions): UseGameWsReturn {
  const [isConnected, setIsConnected] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const clientRef = useRef<Client | null>(null);
  const currentSessionRef = useRef<string | null>(null);
  const everConnectedRef = useRef(false);

  /** 연결 */
  const connect = useCallback((sessionId: string) => {
    if (isConnected || isConnecting) return;

    setIsConnecting(true);
    currentSessionRef.current = sessionId;

    try {
      const socket = new SockJS(WS_BASE_URL);

      const client = new Client({
        webSocketFactory: () => socket as unknown as WebSocket,
        // debug: (str) => console.log('[STOMP Debug]', str),
        reconnectDelay: 5000,
        heartbeatIncoming: 4000,
        heartbeatOutgoing: 4000,
        onConnect: () => {
          console.log('[STOMP] connected');
          everConnectedRef.current = true;
          setIsConnected(true);
          setIsConnecting(false);
          options?.onConnect?.();

          // 게임 채널 구독
          const dest = `/topic/game/${sessionId}`;
          client.subscribe(dest, (message: IMessage) => {
            try {
              const payload = JSON.parse(message.body) as GameWsMessage;
              if (payload.type === 'FEEDBACK' || payload.type === 'LEVEL_DECISION') {
                options?.onFeedback?.(payload);
              }
            } catch (e) {
              console.error('Feedback parse error:', e);
            }
          });
        },
        onDisconnect: () => {
          setIsConnected(false);
          setIsConnecting(false);
          options?.onDisconnect?.();
        },
        onStompError: (frame) => {
          const err = new Error(frame.headers['message'] || 'STOMP error');
          setIsConnected(false);
          setIsConnecting(false);
          options?.onError?.(err);
        },
        onWebSocketClose: (evt) => {
          console.warn('[STOMP] websocket closed', evt.code, evt.reason);
          setIsConnected(false);
          setIsConnecting(false);

          if (!everConnectedRef.current) {
            options?.onError?.(new Error('WebSocket closed before first connect'));
          } else {
            options?.onDisconnect?.();
          }
        },
        onWebSocketError: () => {
          setIsConnected(false);
          setIsConnecting(false);
          if (!everConnectedRef.current) {
            options?.onError?.(new Error('WebSocket error before first connect'));
          } else {
            options?.onDisconnect?.();
          }
        },
      });

      client.activate();
      clientRef.current = client;
    } catch (e) {
      console.error('WS connect error:', e);
      setIsConnecting(false);
      options?.onError?.(e as Error);
    }
  }, [isConnected, isConnecting, options]);

  /** 해제 */
  const disconnect = useCallback(() => {
    if (clientRef.current) {
      clientRef.current.deactivate();
      clientRef.current = null;
    }
    currentSessionRef.current = null;
    setIsConnected(false);
    setIsConnecting(false);
  }, []);

  /** 프레임 전송: /app/game/frame 로 JSON 본문 전송 (기존 이미지 방식) */
  const sendFrame = useCallback(
    async ({ sessionId, blob, currentPlayTime }: { sessionId: string; blob: Blob; currentPlayTime: number }) => {
      const client = clientRef.current;
      if (!client || !client.connected) return;

      const frameData = await blobToBase64(blob);
      const body = JSON.stringify({
        sessionId,
        frameData,            // Base64-encoded-image-string...
        currentPlayTime,      // 초 단위
      });

      // 전송 데이터 로그
      console.log('📤 sendFrame:', {
        sessionId,
        currentPlayTime,
        frameDataLength: frameData.length,
        frameDataPreview: frameData.substring(0, 50) + '...',
      });

      try {
        client.publish({
          destination: '/app/game/frame',
          body,
          headers: { 'content-type': 'application/json' },
        });
      } catch (e) {
        console.error('sendFrame error:', e);
        options?.onError?.(e as Error);
      }
    },
    [options]
  );

  /** Pose 데이터 전송: /app/game/frame 로 poseData 전송 (새로운 방식) */
  const sendPoseData = useCallback(
    ({ sessionId, poseData, currentPlayTime }: { sessionId: string; poseData: number[][]; currentPlayTime: number }) => {
      const client = clientRef.current;
      if (!client || !client.connected) return;

      const body = JSON.stringify({
        sessionId,
        currentPlayTime,
        poseData,  // [[x, y], [x, y], ...] 33개 랜드마크
      });

      // 전송 데이터 로그
      console.log('📤 sendPoseData:', {
        sessionId,
        currentPlayTime,
        landmarkCount: poseData.length,
        sampleLandmarks: poseData.slice(0, 3),  // 처음 3개만 미리보기
      });

      try {
        client.publish({
          destination: '/app/game/frame',
          body,
          headers: { 'content-type': 'application/json' },
        });
      } catch (e) {
        console.error('sendPoseData error:', e);
        options?.onError?.(e as Error);
      }
    },
    [options]
  );

  /** 언마운트 시 정리 */
  useEffect(() => () => disconnect(), [disconnect]);

  return { isConnected, isConnecting, connect, disconnect, sendFrame, sendPoseData, clientRef };
}
