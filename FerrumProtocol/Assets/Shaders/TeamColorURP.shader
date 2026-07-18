Shader "Ferrum Protocol/TeamColorLit"
{
    // Simple URP-forward Lambert shader with a grayscale mask texture that recolors to each
    // player's team color - the mask's white areas take the tint, black areas stay the base
    // texture's original color. Intended for real unit/building art later; placeholder prefabs
    // (Phase 1) use the stock Universal Render Pipeline/Lit shader directly instead.
    Properties
    {
        _BaseMap ("Base Texture", 2D) = "white" {}
        _TeamMaskMap ("Team Color Mask (R)", 2D) = "black" {}
        _TeamColor ("Team Color", Color) = (1,1,1,1)
        _AmbientStrength ("Ambient Strength", Range(0,1)) = 0.25
    }

    SubShader
    {
        Tags { "RenderType" = "Opaque" "RenderPipeline" = "UniversalPipeline" "Queue" = "Geometry" }
        LOD 200

        Pass
        {
            Name "ForwardLit"
            Tags { "LightMode" = "UniversalForward" }

            HLSLPROGRAM
            #pragma vertex vert
            #pragma fragment frag
            #pragma multi_compile _ _MAIN_LIGHT_SHADOWS

            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Core.hlsl"
            #include "Packages/com.unity.render-pipelines.universal/ShaderLibrary/Lighting.hlsl"

            TEXTURE2D(_BaseMap);
            SAMPLER(sampler_BaseMap);
            TEXTURE2D(_TeamMaskMap);
            SAMPLER(sampler_TeamMaskMap);

            CBUFFER_START(UnityPerMaterial)
                float4 _BaseMap_ST;
                float4 _TeamColor;
                float _AmbientStrength;
            CBUFFER_END

            struct Attributes
            {
                float4 positionOS : POSITION;
                float3 normalOS   : NORMAL;
                float2 uv         : TEXCOORD0;
            };

            struct Varyings
            {
                float4 positionHCS : SV_POSITION;
                float2 uv          : TEXCOORD0;
                float3 normalWS    : TEXCOORD1;
            };

            Varyings vert(Attributes IN)
            {
                Varyings OUT;
                VertexPositionInputs positionInputs = GetVertexPositionInputs(IN.positionOS.xyz);
                OUT.positionHCS = positionInputs.positionCS;
                OUT.normalWS = TransformObjectToWorldNormal(IN.normalOS);
                OUT.uv = TRANSFORM_TEX(IN.uv, _BaseMap);
                return OUT;
            }

            half4 frag(Varyings IN) : SV_Target
            {
                half4 baseColor = SAMPLE_TEXTURE2D(_BaseMap, sampler_BaseMap, IN.uv);
                half mask = SAMPLE_TEXTURE2D(_TeamMaskMap, sampler_TeamMaskMap, IN.uv).r;
                half3 albedo = lerp(baseColor.rgb, baseColor.rgb * _TeamColor.rgb, mask);

                Light mainLight = GetMainLight();
                float3 normalWS = normalize(IN.normalWS);
                half nDotL = saturate(dot(normalWS, mainLight.direction));
                half3 lighting = mainLight.color * nDotL + _AmbientStrength;

                return half4(albedo * lighting, baseColor.a);
            }
            ENDHLSL
        }
    }

    FallBack "Universal Render Pipeline/Lit"
}
