Shader "Ferrum Protocol/FogOfWarOverlay"
{
    // World-aligned transparent overlay: samples FogOfWarManager's grid texture (encoded as
    // 0=Unseen, 1=Explored, 2=Visible per FogOfWar/FogOfWarManager.cs) and renders full black
    // over Unseen, dimmed over Explored, and fully clear over Visible. Intended to be applied
    // to a large flat quad positioned above the terrain, sized to match the fog grid's world
    // extents (see Runtime/FogOfWar/FogOfWarRenderer.cs, which keeps the material's world-bounds
    // properties in sync automatically).
    Properties
    {
        _FogTex ("Fog Grid (R: 0/1/2)", 2D) = "black" {}
        _FogWorldMin ("Fog World Min (XZ)", Vector) = (-100, -100, 0, 0)
        _FogWorldSize ("Fog World Size (XZ)", Vector) = (200, 200, 0, 0)
        _ExploredAlpha ("Explored (dimmed) Alpha", Range(0,1)) = 0.55
    }

    SubShader
    {
        Tags { "RenderType" = "Transparent" "RenderPipeline" = "UniversalPipeline" "Queue" = "Transparent" }
        Blend SrcAlpha OneMinusSrcAlpha
        ZWrite Off
        Cull Back

        Pass
        {
            Name "FogOverlay"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"

            TEXTURE2D(_FogTex);
            SAMPLER(sampler_FogTex);

            CBUFFER_START(UnityPerMaterial)
                float4 _FogWorldMin;
                float4 _FogWorldSize;
                float _ExploredAlpha;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float3 positionWS  : TEXCOORD0;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                OUT.positionHCS = positionInputs.positionCS;
                OUT.positionWS = positionInputs.positionWS;
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                float2 uv = (IN.positionWS.xz - _FogWorldMin.xy) / max(_FogWorldSize.xy, 0.0001);
                uv = saturate(uv);

                float rawState = SAMPLE_TEXTURE2D(_FogTex, sampler_FogTex, uv).r * 255.0;
                float state = round(rawState);

                // state 0 = Unseen (opaque black), 1 = Explored (dimmed), 2 = Visible (fully clear).
                float alpha = state < 0.5 ? 1.0 : (state < 1.5 ? _ExploredAlpha : 0.0);
                return half4(0, 0, 0, alpha);
            }
            ENDHLSL
        }
    }
}
