////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

#include <memory>
#include <d3d9.h>
#include <d3d12.h>
#include <DXGIFormat.h>
#include "InteropCommon.h"

#define TRY_D3D(FUNCTION) if (FUNCTION != S_OK) throw gcnew System::Exception(#FUNCTION);

namespace RenderToy
{
	#pragma region - Direct3D12 Enumerations -
	public enum struct DXGIFormat
	{
		Unknown = DXGI_FORMAT_UNKNOWN,
		R32G32B32_Float = DXGI_FORMAT_R32G32B32_FLOAT,
		R8G8B8A8_Unorm = DXGI_FORMAT_R8G8B8A8_UNORM,
		B8G8R8A8_Unorm = DXGI_FORMAT_B8G8R8A8_UNORM,
	};
	public enum struct D3D12Blend
	{
		Zero = D3D12_BLEND_ZERO,
		One = D3D12_BLEND_ONE,
		SrcColor = D3D12_BLEND_SRC_COLOR,
		InvSrcColor = D3D12_BLEND_INV_SRC_COLOR,
		SrcAlpha = D3D12_BLEND_SRC_ALPHA,
		InvSrcAlpha = D3D12_BLEND_INV_SRC_ALPHA,
		DestAlpha = D3D12_BLEND_DEST_ALPHA,
		InvDestAlpha = D3D12_BLEND_INV_DEST_ALPHA,
		DestColor = D3D12_BLEND_DEST_COLOR,
		InvDestColor = D3D12_BLEND_INV_DEST_COLOR,
		SrcAlphaSat = D3D12_BLEND_SRC_ALPHA_SAT,
		BlendFactor = D3D12_BLEND_BLEND_FACTOR,
		InvBlendFactor = D3D12_BLEND_INV_BLEND_FACTOR,
		Src1Color = D3D12_BLEND_SRC1_COLOR,
		InvSrc1Color = D3D12_BLEND_INV_SRC1_COLOR,
		Src1Alpha = D3D12_BLEND_SRC1_ALPHA,
		InvSrc1Alpha = D3D12_BLEND_INV_SRC1_ALPHA,
	};
	public enum struct D3D12BlendOp
	{
		Add = D3D12_BLEND_OP_ADD,
		Subtract = D3D12_BLEND_OP_SUBTRACT,
		RevSubtract = D3D12_BLEND_OP_REV_SUBTRACT,
		Min = D3D12_BLEND_OP_MIN,
		Max = D3D12_BLEND_OP_MAX,
	};
	public enum struct D3D12CPUPageProperty
	{
		Unknown = D3D12_CPU_PAGE_PROPERTY_UNKNOWN,
		NotAvailable = D3D12_CPU_PAGE_PROPERTY_NOT_AVAILABLE,
		WriteCombine = D3D12_CPU_PAGE_PROPERTY_WRITE_COMBINE,
		WriteBack = D3D12_CPU_PAGE_PROPERTY_WRITE_BACK,
	};
	public enum struct D3D12CommandListType
	{
		Direct = D3D12_COMMAND_LIST_TYPE_DIRECT,
		Bundle = D3D12_COMMAND_LIST_TYPE_BUNDLE,
		Compute = D3D12_COMMAND_LIST_TYPE_COMPUTE,
		Copy = D3D12_COMMAND_LIST_TYPE_COPY,
		VideoDecode = D3D12_COMMAND_LIST_TYPE_VIDEO_DECODE,
		VideoProcess = D3D12_COMMAND_LIST_TYPE_VIDEO_PROCESS,
	};
	public enum struct D3D12CommandQueueFlags
	{
		None = D3D12_COMMAND_QUEUE_FLAG_NONE,
		DisableGpuTimeout = D3D12_COMMAND_QUEUE_FLAG_DISABLE_GPU_TIMEOUT,
	};
	public enum struct D3D12ComparisonFunc
	{
		Never = D3D12_COMPARISON_FUNC_NEVER,
		Less = D3D12_COMPARISON_FUNC_LESS,
		Equal = D3D12_COMPARISON_FUNC_EQUAL,
		LessEqual = D3D12_COMPARISON_FUNC_LESS_EQUAL,
		Greater = D3D12_COMPARISON_FUNC_GREATER,
		NotEqual = D3D12_COMPARISON_FUNC_NOT_EQUAL,
		GreaterEqual = D3D12_COMPARISON_FUNC_GREATER_EQUAL,
		Always = D3D12_COMPARISON_FUNC_ALWAYS,
	};
	public enum struct D3D12ConservativeRasterizationMode
	{
		Off = D3D12_CONSERVATIVE_RASTERIZATION_MODE_OFF,
		On = D3D12_CONSERVATIVE_RASTERIZATION_MODE_ON,
	};
	public enum struct D3D12CullMode
	{
		None = D3D12_CULL_MODE_NONE,
		Front = D3D12_CULL_MODE_FRONT,
		Back = D3D12_CULL_MODE_BACK,
	};
	public enum struct D3D12DepthWriteMask
	{
		Zero = D3D12_DEPTH_WRITE_MASK_ZERO,
		All = D3D12_DEPTH_WRITE_MASK_ALL,
	};
	public enum struct D3D12DescriptorHeapFlags {
		None = D3D12_DESCRIPTOR_HEAP_FLAG_NONE,
		ShaderVisible = D3D12_DESCRIPTOR_HEAP_FLAG_SHADER_VISIBLE,
	};
	public enum struct D3D12DescriptorHeapType
	{
		CbvSrvUav = D3D12_DESCRIPTOR_HEAP_TYPE_CBV_SRV_UAV,
		Sampler = D3D12_DESCRIPTOR_HEAP_TYPE_SAMPLER,
		Rtv = D3D12_DESCRIPTOR_HEAP_TYPE_RTV,
		Dsv = D3D12_DESCRIPTOR_HEAP_TYPE_DSV,
		NumTypes = D3D12_DESCRIPTOR_HEAP_TYPE_NUM_TYPES,
	};
	public enum struct D3D12FenceFlags
	{
		None = D3D12_FENCE_FLAG_NONE,
		Shared = D3D12_FENCE_FLAG_SHARED,
		SharedCrossAdapter = D3D12_FENCE_FLAG_SHARED_CROSS_ADAPTER,
		NonMonitored = D3D12_FENCE_FLAG_NON_MONITORED,
	};
	public enum struct D3D12FillMode
	{
		Wireframe = D3D12_FILL_MODE_WIREFRAME,
		Solid = D3D12_FILL_MODE_SOLID,
	};
	public enum struct D3D12HeapFlags
	{
		None = D3D12_HEAP_FLAG_NONE,
		Shared = D3D12_HEAP_FLAG_SHARED,
		DenyBuffers = D3D12_HEAP_FLAG_DENY_BUFFERS,
		AllowDisplay = D3D12_HEAP_FLAG_ALLOW_DISPLAY,
		SharedCrossAdapter = D3D12_HEAP_FLAG_SHARED_CROSS_ADAPTER,
		DenyRTDSTextures = D3D12_HEAP_FLAG_DENY_RT_DS_TEXTURES,
		DenyNonRTDSTextures = D3D12_HEAP_FLAG_DENY_NON_RT_DS_TEXTURES,
		HardwareProtected = D3D12_HEAP_FLAG_HARDWARE_PROTECTED,
		AllowWriteWatch = D3D12_HEAP_FLAG_ALLOW_WRITE_WATCH,
		AllowAllBuffersAndTextures = D3D12_HEAP_FLAG_ALLOW_ALL_BUFFERS_AND_TEXTURES,
		AllowOnlyBuffers = D3D12_HEAP_FLAG_ALLOW_ONLY_BUFFERS,
		AllowOnlyNonRTDSTextures = D3D12_HEAP_FLAG_ALLOW_ONLY_NON_RT_DS_TEXTURES,
		AllowOnlyRTDSTextures = D3D12_HEAP_FLAG_ALLOW_ONLY_RT_DS_TEXTURES,
	};
	public enum struct D3D12HeapType
	{
		Default = D3D12_HEAP_TYPE_DEFAULT,
		Upload = D3D12_HEAP_TYPE_UPLOAD,
		Readback = D3D12_HEAP_TYPE_READBACK,
		Custom = D3D12_HEAP_TYPE_CUSTOM,
	};
	public enum struct D3D12IndexBufferStripCutValue
	{
		Disabled = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_DISABLED,
		CutFFFF = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFF,
		CutFFFFFFFFF = D3D12_INDEX_BUFFER_STRIP_CUT_VALUE_0xFFFFFFFF,
	};
	public enum struct D3D12InputClassification
	{
		PerVertexData = D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
		PerInstanceData = D3D12_INPUT_CLASSIFICATION_PER_INSTANCE_DATA,
	};
	public enum struct D3D12LogicOp
	{
		Clear = D3D12_LOGIC_OP_CLEAR,
		Set = D3D12_LOGIC_OP_SET,
		Copy = D3D12_LOGIC_OP_COPY,
		CopyInverted = D3D12_LOGIC_OP_COPY_INVERTED,
		NoOp = D3D12_LOGIC_OP_NOOP,
		Invert = D3D12_LOGIC_OP_INVERT,
		And = D3D12_LOGIC_OP_AND,
		Nand = D3D12_LOGIC_OP_NAND,
		Or = D3D12_LOGIC_OP_OR,
		Nor = D3D12_LOGIC_OP_NOR,
		Xor = D3D12_LOGIC_OP_XOR,
		Equiv = D3D12_LOGIC_OP_EQUIV,
		AndReverse = D3D12_LOGIC_OP_AND_REVERSE,
		AndInverted = D3D12_LOGIC_OP_AND_INVERTED,
		OrReverse = D3D12_LOGIC_OP_OR_REVERSE,
		OrInverted = D3D12_LOGIC_OP_OR_INVERTED,
	};
	public enum struct D3D12MemoryPool
	{
		Unknown = D3D12_MEMORY_POOL_UNKNOWN,
		L0 = D3D12_MEMORY_POOL_L0,
		L1 = D3D12_MEMORY_POOL_L1,
	};
	public enum struct D3D12PipelineStateFlags
	{
		None = D3D12_PIPELINE_STATE_FLAG_NONE,
		ToolDebug = D3D12_PIPELINE_STATE_FLAG_TOOL_DEBUG,
	};
	public enum struct D3D12PrimitiveTopology
	{
		Undefined = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED,
		PointList = D3D_PRIMITIVE_TOPOLOGY_POINTLIST,
		LineList = D3D_PRIMITIVE_TOPOLOGY_LINELIST,
		LineStrip = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP,
		TriangleList = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
		TriangleStrip = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
		LineListAdj = D3D_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,
		LineStripAdj = D3D_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ,
		TriangleListAdj = D3D_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ,
		TriangleStripAdj = D3D_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ,
		PatchList1 = D3D_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST,
		PatchList2 = D3D_PRIMITIVE_TOPOLOGY_2_CONTROL_POINT_PATCHLIST,
		PatchList3 = D3D_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST,
		PatchList4 = D3D_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST,
		PatchList5 = D3D_PRIMITIVE_TOPOLOGY_5_CONTROL_POINT_PATCHLIST,
		PatchList6 = D3D_PRIMITIVE_TOPOLOGY_6_CONTROL_POINT_PATCHLIST,
		PatchList7 = D3D_PRIMITIVE_TOPOLOGY_7_CONTROL_POINT_PATCHLIST,
		PatchList8 = D3D_PRIMITIVE_TOPOLOGY_8_CONTROL_POINT_PATCHLIST,
		PatchList9 = D3D_PRIMITIVE_TOPOLOGY_9_CONTROL_POINT_PATCHLIST,
		PatchList10 = D3D_PRIMITIVE_TOPOLOGY_10_CONTROL_POINT_PATCHLIST,
		PatchList11 = D3D_PRIMITIVE_TOPOLOGY_11_CONTROL_POINT_PATCHLIST,
		PatchList12 = D3D_PRIMITIVE_TOPOLOGY_12_CONTROL_POINT_PATCHLIST,
		PatchList13 = D3D_PRIMITIVE_TOPOLOGY_13_CONTROL_POINT_PATCHLIST,
		PatchList14 = D3D_PRIMITIVE_TOPOLOGY_14_CONTROL_POINT_PATCHLIST,
		PatchList15 = D3D_PRIMITIVE_TOPOLOGY_15_CONTROL_POINT_PATCHLIST,
		PatchList16 = D3D_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST,
		PatchList17 = D3D_PRIMITIVE_TOPOLOGY_17_CONTROL_POINT_PATCHLIST,
		PatchList18 = D3D_PRIMITIVE_TOPOLOGY_18_CONTROL_POINT_PATCHLIST,
		PatchList19 = D3D_PRIMITIVE_TOPOLOGY_19_CONTROL_POINT_PATCHLIST,
		PatchList20 = D3D_PRIMITIVE_TOPOLOGY_20_CONTROL_POINT_PATCHLIST,
		PatchList21 = D3D_PRIMITIVE_TOPOLOGY_21_CONTROL_POINT_PATCHLIST,
		PatchList22 = D3D_PRIMITIVE_TOPOLOGY_22_CONTROL_POINT_PATCHLIST,
		PatchList23 = D3D_PRIMITIVE_TOPOLOGY_23_CONTROL_POINT_PATCHLIST,
		PatchList24 = D3D_PRIMITIVE_TOPOLOGY_24_CONTROL_POINT_PATCHLIST,
		PatchList25 = D3D_PRIMITIVE_TOPOLOGY_25_CONTROL_POINT_PATCHLIST,
		PatchList26 = D3D_PRIMITIVE_TOPOLOGY_26_CONTROL_POINT_PATCHLIST,
		PatchList27 = D3D_PRIMITIVE_TOPOLOGY_27_CONTROL_POINT_PATCHLIST,
		PatchList28 = D3D_PRIMITIVE_TOPOLOGY_28_CONTROL_POINT_PATCHLIST,
		PatchList29 = D3D_PRIMITIVE_TOPOLOGY_29_CONTROL_POINT_PATCHLIST,
		PatchList30 = D3D_PRIMITIVE_TOPOLOGY_30_CONTROL_POINT_PATCHLIST,
		PatchList31 = D3D_PRIMITIVE_TOPOLOGY_31_CONTROL_POINT_PATCHLIST,
		PatchList32 = D3D_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST,
		D3D10Undefined = D3D_PRIMITIVE_TOPOLOGY_UNDEFINED,
		D3D10PointList = D3D10_PRIMITIVE_TOPOLOGY_POINTLIST,
		D3D10LineList = D3D10_PRIMITIVE_TOPOLOGY_LINELIST,
		D3D10LineStrip = D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP,
		D3D10TriangleList = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
		D3D10TriangleStrip = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
		D3D10LineListAdj = D3D10_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,
		D3D10LineStripAdj = D3D10_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ,
		D3D10TriangleListAdj = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ,
		D3D10TriangleStripAdj = D3D10_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ,
		D3D11Unknown = D3D11_PRIMITIVE_TOPOLOGY_UNDEFINED,
		D3D11PointList = D3D11_PRIMITIVE_TOPOLOGY_POINTLIST,
		D3D11LineList = D3D11_PRIMITIVE_TOPOLOGY_LINELIST,
		D3D11LineStrip = D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP,
		D3D11TriangleList = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST,
		D3D11TriangleStrip = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP,
		D3D11LineListAdj = D3D11_PRIMITIVE_TOPOLOGY_LINELIST_ADJ,
		D3D11LineStripAdj = D3D11_PRIMITIVE_TOPOLOGY_LINESTRIP_ADJ,
		D3D11TriangleListAdj = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLELIST_ADJ,
		D3D11TriangleStripAdj = D3D11_PRIMITIVE_TOPOLOGY_TRIANGLESTRIP_ADJ,
		D3D11PatchList1 = D3D11_PRIMITIVE_TOPOLOGY_1_CONTROL_POINT_PATCHLIST,
		D3D11PatchList2 = D3D11_PRIMITIVE_TOPOLOGY_2_CONTROL_POINT_PATCHLIST,
		D3D11PatchList3 = D3D11_PRIMITIVE_TOPOLOGY_3_CONTROL_POINT_PATCHLIST,
		D3D11PatchList4 = D3D11_PRIMITIVE_TOPOLOGY_4_CONTROL_POINT_PATCHLIST,
		D3D11PatchList5 = D3D11_PRIMITIVE_TOPOLOGY_5_CONTROL_POINT_PATCHLIST,
		D3D11PatchList6 = D3D11_PRIMITIVE_TOPOLOGY_6_CONTROL_POINT_PATCHLIST,
		D3D11PatchList7 = D3D11_PRIMITIVE_TOPOLOGY_7_CONTROL_POINT_PATCHLIST,
		D3D11PatchList8 = D3D11_PRIMITIVE_TOPOLOGY_8_CONTROL_POINT_PATCHLIST,
		D3D11PatchList9 = D3D11_PRIMITIVE_TOPOLOGY_9_CONTROL_POINT_PATCHLIST,
		D3D11PatchList10 = D3D11_PRIMITIVE_TOPOLOGY_10_CONTROL_POINT_PATCHLIST,
		D3D11PatchList11 = D3D11_PRIMITIVE_TOPOLOGY_11_CONTROL_POINT_PATCHLIST,
		D3D11PatchList12 = D3D11_PRIMITIVE_TOPOLOGY_12_CONTROL_POINT_PATCHLIST,
		D3D11PatchList13 = D3D11_PRIMITIVE_TOPOLOGY_13_CONTROL_POINT_PATCHLIST,
		D3D11PatchList14 = D3D11_PRIMITIVE_TOPOLOGY_14_CONTROL_POINT_PATCHLIST,
		D3D11PatchList15 = D3D11_PRIMITIVE_TOPOLOGY_15_CONTROL_POINT_PATCHLIST,
		D3D11PatchList16 = D3D11_PRIMITIVE_TOPOLOGY_16_CONTROL_POINT_PATCHLIST,
		D3D11PatchList17 = D3D11_PRIMITIVE_TOPOLOGY_17_CONTROL_POINT_PATCHLIST,
		D3D11PatchList18 = D3D11_PRIMITIVE_TOPOLOGY_18_CONTROL_POINT_PATCHLIST,
		D3D11PatchList19 = D3D11_PRIMITIVE_TOPOLOGY_19_CONTROL_POINT_PATCHLIST,
		D3D11PatchList20 = D3D11_PRIMITIVE_TOPOLOGY_20_CONTROL_POINT_PATCHLIST,
		D3D11PatchList21 = D3D11_PRIMITIVE_TOPOLOGY_21_CONTROL_POINT_PATCHLIST,
		D3D11PatchList22 = D3D11_PRIMITIVE_TOPOLOGY_22_CONTROL_POINT_PATCHLIST,
		D3D11PatchList23 = D3D11_PRIMITIVE_TOPOLOGY_23_CONTROL_POINT_PATCHLIST,
		D3D11PatchList24 = D3D11_PRIMITIVE_TOPOLOGY_24_CONTROL_POINT_PATCHLIST,
		D3D11PatchList25 = D3D11_PRIMITIVE_TOPOLOGY_25_CONTROL_POINT_PATCHLIST,
		D3D11PatchList26 = D3D11_PRIMITIVE_TOPOLOGY_26_CONTROL_POINT_PATCHLIST,
		D3D11PatchList27 = D3D11_PRIMITIVE_TOPOLOGY_27_CONTROL_POINT_PATCHLIST,
		D3D11PatchList28 = D3D11_PRIMITIVE_TOPOLOGY_28_CONTROL_POINT_PATCHLIST,
		D3D11PatchList29 = D3D11_PRIMITIVE_TOPOLOGY_29_CONTROL_POINT_PATCHLIST,
		D3D11PatchList30 = D3D11_PRIMITIVE_TOPOLOGY_30_CONTROL_POINT_PATCHLIST,
		D3D11PatchList31 = D3D11_PRIMITIVE_TOPOLOGY_31_CONTROL_POINT_PATCHLIST,
		D3D11PatchList32 = D3D11_PRIMITIVE_TOPOLOGY_32_CONTROL_POINT_PATCHLIST,
	};
	public enum struct D3D12PrimitiveTopologyType
	{
		Undefined = D3D12_PRIMITIVE_TOPOLOGY_TYPE_UNDEFINED,
		Point = D3D12_PRIMITIVE_TOPOLOGY_TYPE_POINT,
		Line = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE,
		Triangle = D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE,
		Patch = D3D12_PRIMITIVE_TOPOLOGY_TYPE_PATCH,
	};
	public enum struct D3D12ResourceBarrierFlags
	{
		None = D3D12_RESOURCE_BARRIER_FLAG_NONE,
		BeginOnly = D3D12_RESOURCE_BARRIER_FLAG_BEGIN_ONLY,
		EndOnly = D3D12_RESOURCE_BARRIER_FLAG_END_ONLY,
	};
	public enum struct D3D12ResourceBarrierType
	{
		Transition = D3D12_RESOURCE_BARRIER_TYPE_TRANSITION,
		Aliasing = D3D12_RESOURCE_BARRIER_TYPE_ALIASING,
		UAV = D3D12_RESOURCE_BARRIER_TYPE_UAV,
	};
	public enum struct D3D12ResourceDimension
	{
		Unknown = D3D12_RESOURCE_DIMENSION_UNKNOWN,
		Buffer = D3D12_RESOURCE_DIMENSION_BUFFER,
		Texture1D = D3D12_RESOURCE_DIMENSION_TEXTURE1D,
		Texture2D = D3D12_RESOURCE_DIMENSION_TEXTURE2D,
		Texture3D = D3D12_RESOURCE_DIMENSION_TEXTURE3D,
	};
	public enum struct D3D12ResourceFlags
	{
		None = D3D12_RESOURCE_FLAG_NONE,
		AllowRenderTarget = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET,
		AllowDepthStencil = D3D12_RESOURCE_FLAG_ALLOW_DEPTH_STENCIL,
		AllowUnorderedAccess = D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS,
		DenyShaderResource = D3D12_RESOURCE_FLAG_DENY_SHADER_RESOURCE,
		AllowCrossAdapter = D3D12_RESOURCE_FLAG_ALLOW_CROSS_ADAPTER,
		AllowSimultaneousAccess = D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
		VideoDecodeReferenceOnly = D3D12_RESOURCE_FLAG_VIDEO_DECODE_REFERENCE_ONLY,
	};
	public enum struct D3D12ResourceStates
	{
		Common = D3D12_RESOURCE_STATE_COMMON,
		VertexAndConstantBuffer = D3D12_RESOURCE_STATE_VERTEX_AND_CONSTANT_BUFFER,
		IndexBuffer = D3D12_RESOURCE_STATE_INDEX_BUFFER,
		RenderTarget = D3D12_RESOURCE_STATE_RENDER_TARGET,
		UnordererdAccess = D3D12_RESOURCE_STATE_UNORDERED_ACCESS,
		DepthWrite = D3D12_RESOURCE_STATE_DEPTH_WRITE,
		DepthRead = D3D12_RESOURCE_STATE_DEPTH_READ,
		NonPixelShaderResource = D3D12_RESOURCE_STATE_NON_PIXEL_SHADER_RESOURCE,
		PixelShaderResource = D3D12_RESOURCE_STATE_PIXEL_SHADER_RESOURCE,
		StreamOut = D3D12_RESOURCE_STATE_STREAM_OUT,
		IndirectArgument = D3D12_RESOURCE_STATE_INDIRECT_ARGUMENT,
		CopyDest = D3D12_RESOURCE_STATE_COPY_DEST,
		CopySource = D3D12_RESOURCE_STATE_COPY_SOURCE,
		ResolveDest = D3D12_RESOURCE_STATE_RESOLVE_DEST,
		ResolveSource = D3D12_RESOURCE_STATE_RESOLVE_SOURCE,
		GenericRead = D3D12_RESOURCE_STATE_GENERIC_READ,
		Present = D3D12_RESOURCE_STATE_PRESENT,
		Predication = D3D12_RESOURCE_STATE_PREDICATION,
		VideoDecodeRead = D3D12_RESOURCE_STATE_VIDEO_DECODE_READ,
		VideoDecodeWrite = D3D12_RESOURCE_STATE_VIDEO_DECODE_WRITE,
		VideoProcessRead = D3D12_RESOURCE_STATE_VIDEO_PROCESS_READ,
		VideoProcessWrite = D3D12_RESOURCE_STATE_VIDEO_PROCESS_WRITE,
	};
	public enum struct D3D12RtvDimension
	{
		Unknown = D3D12_RTV_DIMENSION_UNKNOWN,
		Buffer = D3D12_RTV_DIMENSION_BUFFER,
		Texture1D = D3D12_RTV_DIMENSION_TEXTURE1D,
		Texture1DArray = D3D12_RTV_DIMENSION_TEXTURE1DARRAY,
		Texture2D = D3D12_RTV_DIMENSION_TEXTURE2D,
		Texture2DArray = D3D12_RTV_DIMENSION_TEXTURE2DARRAY,
		Texture2DMS = D3D12_RTV_DIMENSION_TEXTURE2DMS,
		Texture2DMSArray = D3D12_RTV_DIMENSION_TEXTURE2DMSARRAY,
		Texture3D = D3D12_RTV_DIMENSION_TEXTURE3D,
	};
	public enum struct D3D12StencilOp
	{
		Keep = D3D12_STENCIL_OP_KEEP,
		Zero = D3D12_STENCIL_OP_ZERO,
		Replace = D3D12_STENCIL_OP_REPLACE,
		IncrSat = D3D12_STENCIL_OP_INCR_SAT,
		DecrSat = D3D12_STENCIL_OP_DECR_SAT,
		Invert = D3D12_STENCIL_OP_INVERT,
		Incr = D3D12_STENCIL_OP_INCR,
		Decr = D3D12_STENCIL_OP_DECR,
	};
	public enum struct D3D12TextureLayout
	{
		Unknown = D3D12_TEXTURE_LAYOUT_UNKNOWN,
		RowMajor = D3D12_TEXTURE_LAYOUT_ROW_MAJOR,
		UndefinedSwizzle = D3D12_TEXTURE_LAYOUT_64KB_UNDEFINED_SWIZZLE,
		StandardSwizzle = D3D12_TEXTURE_LAYOUT_64KB_STANDARD_SWIZZLE,
	};
	#pragma endregion
	#pragma region - Direct3D12 Structures -
	public value struct D3D12CommandQueueDesc
	{
		D3D12CommandListType	Type;
		INT                     Priority;
		D3D12CommandQueueFlags	Flags;
		UINT                    NodeMask;
	};
	public value struct D3D12DepthStencilOpDesc
	{
		D3D12StencilOp      StencilFailOp;
		D3D12StencilOp      StencilDepthFailOp;
		D3D12StencilOp      StencilPassOp;
		D3D12ComparisonFunc StencilFunc;
	};
	public value struct D3D12DepthStencilDesc
	{
		BOOL                    DepthEnable;
		D3D12DepthWriteMask     DepthWriteMask;
		D3D12ComparisonFunc     DepthFunc;
		BOOL                    StencilEnable;
		UINT8                   StencilReadMask;
		UINT8                   StencilWriteMask;
		D3D12DepthStencilOpDesc	FrontFace;
		D3D12DepthStencilOpDesc	BackFace;
	};
	public value struct D3D12RasterizerDesc
	{
		D3D12FillMode                       FillMode;
		D3D12CullMode                       CullMode;
		BOOL                                FrontCounterClockwise;
		INT                                 DepthBias;
		FLOAT                               DepthBiasClamp;
		FLOAT                               SlopeScaledDepthBias;
		BOOL                                DepthClipEnable;
		BOOL                                MultisampleEnable;
		BOOL                                AntialiasedLineEnable;
		UINT                                ForcedSampleCount;
		D3D12ConservativeRasterizationMode	ConservativeRaster;
	};
	public value struct D3D12RenderTargetBlendDesc
	{
		BOOL			BlendEnable;
		BOOL			LogicOpEnable;
		D3D12Blend		SrcBlend;
		D3D12Blend		DestBlend;
		D3D12BlendOp	BlendOp;
		D3D12Blend		SrcBlendAlpha;
		D3D12Blend		DestBlendAlpha;
		D3D12BlendOp    BlendOpAlpha;
		D3D12LogicOp	LogicOp;
		UINT8			RenderTargetWriteMask;
	};
	public value struct D3D12Tex2DRtv
	{
		UINT MipSlice;
		UINT PlaneSlice;
	};
	public value struct D3D12RenderTargetViewDesc
	{
		DXGIFormat			Format;
		D3D12RtvDimension	ViewDimension;
		D3D12Tex2DRtv		Texture2D;
	};
	ref class D3D12Resource;
	public value struct D3D12ResourceTransitionBarrier
	{
		D3D12Resource^			pResource;
		UINT					Subresource;
		D3D12ResourceStates		StateBefore;
		D3D12ResourceStates		StateAfter;
	};
	public value struct D3D12ResourceBarrier
	{
		D3D12ResourceBarrierType Type;
		D3D12ResourceBarrierFlags Flags;
		D3D12ResourceTransitionBarrier Transition;
	};
	public value struct D3D12BlendDesc
	{
		BOOL                        AlphaToCoverageEnable;
		BOOL						IndependentBlendEnable;
		D3D12RenderTargetBlendDesc	RenderTarget0;
		D3D12RenderTargetBlendDesc	RenderTarget1;
		D3D12RenderTargetBlendDesc	RenderTarget2;
		D3D12RenderTargetBlendDesc	RenderTarget3;
		D3D12RenderTargetBlendDesc	RenderTarget4;
		D3D12RenderTargetBlendDesc	RenderTarget5;
		D3D12RenderTargetBlendDesc	RenderTarget6;
		D3D12RenderTargetBlendDesc	RenderTarget7;
	};
	public value struct D3D12CachedPipelineState
	{
		cli::array<byte>^	pCachedBlob;
	};
	public value struct D3D12ClearValue
	{
		DXGIFormat Format;
		float R, G, B, A;
	};
	public value struct D3D12CPUDescriptorHandle
	{
		System::IntPtr ptr;
	};
	public value struct D3D12DescriptorHeapDesc
	{
		D3D12DescriptorHeapType		Type;
		UINT                        NumDescriptors;
		D3D12DescriptorHeapFlags	Flags;
		UINT                        NodeMask;
	};
	public value struct D3D12InputElementDesc
	{
		System::String^				SemanticName;
		UINT						SemanticIndex;
		DXGIFormat					Format;
		UINT						InputSlot;
		UINT						AlignedByteOffset;
		D3D12InputClassification	InputSlotClass;
		UINT						InstanceDataStepRate;
	};
	public value struct D3D12InputLayoutDesc
	{
		cli::array<D3D12InputElementDesc>^	pInputElementDescs;
	};
	public value struct DXGISampleDesc
	{
		UINT Count;
		UINT Quality;
	};
	public value struct D3D12HeapProperties
	{
		D3D12HeapType			Type;
		D3D12CPUPageProperty	CPUPageProperty;
		D3D12MemoryPool			MemoryPoolPreference;
		UINT					CreationNodeMask;
		UINT					VisibleNodeMask;
	};
	public value struct D3D12SODeclarationEntry
	{
		UINT			Stream;
		System::String^ SemanticName;
		UINT			SemanticIndex;
		BYTE			StartComponent;
		BYTE			ComponentCount;
		BYTE			OutputSlot;
	};
	public value struct D3D12StreamOutputDesc
	{
		cli::array<D3D12SODeclarationEntry>^	pSODeclaration;
		cli::array<UINT>^						pBufferStrides;
		UINT									RasterizedStream;
	};
	ref class D3D12RootSignature;
	public value struct D3D12GraphicsPipelineStateDesc
	{
		D3D12RootSignature^				pRootSignature;
		cli::array<byte>^				VS;
		cli::array<byte>^				PS;
		cli::array<byte>^				DS;
		cli::array<byte>^				HS;
		cli::array<byte>^				GS;
		D3D12StreamOutputDesc			StreamOutput;
		D3D12BlendDesc					BlendState;
		UINT							SampleMask;
		D3D12RasterizerDesc				RasterizerState;
		D3D12DepthStencilDesc			DepthStencilState;
		D3D12InputLayoutDesc			InputLayout;
		D3D12IndexBufferStripCutValue	IBStripCutValue;
		D3D12PrimitiveTopologyType      PrimitiveTopologyType;
		UINT                            NumRenderTargets;
		DXGIFormat                      RTVFormats0;
		DXGIFormat                      RTVFormats1;
		DXGIFormat                      RTVFormats2;
		DXGIFormat                      RTVFormats3;
		DXGIFormat                      RTVFormats4;
		DXGIFormat                      RTVFormats5;
		DXGIFormat                      RTVFormats6;
		DXGIFormat                      RTVFormats7;
		DXGIFormat                      DSVFormat;
		DXGISampleDesc                  SampleDesc;
		UINT                            NodeMask;
		D3D12CachedPipelineState		CachedPSO;
		D3D12PipelineStateFlags			Flags;
	};
	public value struct D3D12Rect
	{
		LONG    left;
		LONG    top;
		LONG    right;
		LONG    bottom;
	};
	public value struct D3D12ResourceDesc
	{
		D3D12ResourceDimension Dimension;
		UINT64                 Alignment;
		UINT64                 Width;
		UINT                   Height;
		UINT16                 DepthOrArraySize;
		UINT16                 MipLevels;
		DXGIFormat             Format;
		DXGISampleDesc         SampleDesc;
		D3D12TextureLayout     Layout;
		D3D12ResourceFlags     Flags;
	};
	public value struct D3D12VertexBufferView
	{
		D3D12_GPU_VIRTUAL_ADDRESS	BufferLocation;
		UINT						SizeInBytes;
		UINT						StrideInBytes;
	};
	public value struct D3D12Viewport
	{
		FLOAT TopLeftX;
		FLOAT TopLeftY;
		FLOAT Width;
		FLOAT Height;
		FLOAT MinDepth;
		FLOAT MaxDepth;
	};
	#pragma endregion
	#pragma region - D3D12CommandAllocator -
	public ref class D3D12CommandAllocator : public COMWrapper<ID3D12CommandAllocator>
	{
	public:
		D3D12CommandAllocator(ID3D12CommandAllocator *obj) : COMWrapper(obj)
		{
		}
		void Reset()
		{
			TRY_D3D(pWrapped->Reset());
		}
	};
	#pragma endregion
	#pragma region - D3D12DescriptorHeap -
	public ref class D3D12DescriptorHeap : public COMWrapper<ID3D12DescriptorHeap>
	{
	public:
		D3D12DescriptorHeap(ID3D12DescriptorHeap *pObj) : COMWrapper(pObj)
		{
		}
		D3D12CPUDescriptorHandle GetCPUDescriptorHandleForHeapStart()
		{
			D3D12_CPU_DESCRIPTOR_HANDLE result = pWrapped->GetCPUDescriptorHandleForHeapStart();
			D3D12CPUDescriptorHandle convert;
			convert.ptr = System::IntPtr((void*)result.ptr);
			return convert;
		}
	};
	#pragma endregion
	#pragma region - D3D12Fence -
	public ref class D3D12Fence : public COMWrapper<ID3D12Fence>
	{
	public:
		D3D12Fence(ID3D12Fence *pObj) : COMWrapper(pObj)
		{
		}
		void SetEventOnCompletion(UINT64 Value, System::IntPtr hEvent)
		{
			TRY_D3D(pWrapped->SetEventOnCompletion(Value, hEvent.ToPointer()));
		}
	};
	#pragma endregion
	#pragma region - D3D12PipelineState -
	public ref class D3D12PipelineState : public COMWrapper<ID3D12PipelineState>
	{
	public:
		D3D12PipelineState(ID3D12PipelineState *pObj) : COMWrapper(pObj)
		{
		}
	};
	#pragma endregion
	#pragma region - D3D12Resource -
	public ref class D3D12Resource : public COMWrapper<ID3D12Resource>
	{
	public:
		D3D12Resource(ID3D12Resource *obj) : COMWrapper(obj)
		{
		}
		D3D12_GPU_VIRTUAL_ADDRESS GetGPUVirtualAddress()
		{
			return pWrapped->GetGPUVirtualAddress();
		}
		System::IntPtr Map(UINT Subresource)
		{
			void *ppData = nullptr;
			TRY_D3D(pWrapped->Map(Subresource, nullptr, &ppData));
			return System::IntPtr(ppData);
		}
		void ReadFromSubresource(System::IntPtr pDstData, UINT DstRowPitch, UINT DstDepthPitch, UINT SrcSubresource)
		{
			TRY_D3D(pWrapped->ReadFromSubresource(pDstData.ToPointer(), DstRowPitch, DstDepthPitch, SrcSubresource, nullptr));
		}
		void Unmap(UINT Subresource)
		{
			pWrapped->Unmap(Subresource, nullptr);
		}
	};
	#pragma endregion
	#pragma region - D3D12RootSignature -
	public ref class D3D12RootSignature : public COMWrapper<ID3D12RootSignature>
	{
	public:
		D3D12RootSignature(ID3D12RootSignature *pObj) : COMWrapper(pObj)
		{
		}
	};
	#pragma endregion
	#pragma region - D3D12GraphicsCommandList1 -
	public ref class D3D12GraphicsCommandList1 : public COMWrapper<ID3D12GraphicsCommandList1>
	{
	public:
		D3D12GraphicsCommandList1(ID3D12GraphicsCommandList1 *pObj) : COMWrapper(pObj)
		{
		}
		void ClearRenderTargetView(D3D12CPUDescriptorHandle RenderTargetView, float R, float G, float B, float A)
		{
			float rgba[4] = { R, G, B, A };
			D3D12_CPU_DESCRIPTOR_HANDLE desc;
			desc.ptr = (SIZE_T)RenderTargetView.ptr.ToPointer();
			pWrapped->ClearRenderTargetView(desc, rgba, 0U, nullptr);
		};
		void Close()
		{
			TRY_D3D(pWrapped->Close());
		}
		void DrawInstanced(UINT VertexCountPerInstance, UINT InstanceCount, UINT StartVertexLocation, UINT StartInstanceLocation)
		{
			pWrapped->DrawInstanced(VertexCountPerInstance, InstanceCount, StartVertexLocation, StartInstanceLocation);
		}
		void IASetPrimitiveTopology(D3D12PrimitiveTopology PrimitiveTopology)
		{
			pWrapped->IASetPrimitiveTopology((D3D12_PRIMITIVE_TOPOLOGY)PrimitiveTopology);
		}
		void IASetVertexBuffers(UINT StartSlot, cli::array<D3D12VertexBufferView> ^pViews)
		{
			pin_ptr<D3D12VertexBufferView> pViewsM = &pViews[0];
			pWrapped->IASetVertexBuffers(StartSlot, pViews->Length, (D3D12_VERTEX_BUFFER_VIEW*)&pViewsM[0]);
		}
		void OMSetRenderTargets(cli::array<D3D12CPUDescriptorHandle> ^pRenderTargetDescriptors, BOOL RTsSingleHandleToDescriptorRange, System::Nullable<D3D12CPUDescriptorHandle> pDepthStencilDescriptor)
		{
			std::unique_ptr<D3D12_CPU_DESCRIPTOR_HANDLE[]> pRenderTargetDescriptorsM(new D3D12_CPU_DESCRIPTOR_HANDLE[pRenderTargetDescriptors->Length]);
			for (int i = 0; i < pRenderTargetDescriptors->Length; ++i)
			{
				pRenderTargetDescriptorsM[i].ptr = (SIZE_T)pRenderTargetDescriptors[i].ptr.ToPointer();
			}
			D3D12_CPU_DESCRIPTOR_HANDLE pDepthStencilDescriptorM;
			pDepthStencilDescriptorM.ptr = (SIZE_T)(pDepthStencilDescriptor.HasValue ? pDepthStencilDescriptor.Value.ptr.ToPointer() : nullptr);
			pWrapped->OMSetRenderTargets(pRenderTargetDescriptors->Length, pRenderTargetDescriptorsM.get(), RTsSingleHandleToDescriptorRange, pDepthStencilDescriptor.HasValue ? &pDepthStencilDescriptorM : nullptr);
		}
		void RSSetScissorRects(cli::array<D3D12Rect> ^pRects)
		{
			pin_ptr<D3D12Rect> pRectsM = &pRects[0];
			pWrapped->RSSetScissorRects(pRects->Length, reinterpret_cast<D3D12_RECT*>(&pRectsM[0]));
		}
		void RSSetViewports(cli::array<D3D12Viewport> ^pViewports)
		{
			pin_ptr<D3D12Viewport> pViewportsM = &pViewports[0];
			pWrapped->RSSetViewports(pViewports->Length, reinterpret_cast<D3D12_VIEWPORT*>(&pViewportsM[0]));
		}
		void SetDescriptorHeaps(cli::array<D3D12DescriptorHeap^> ^ppDescriptorHeaps)
		{
			std::unique_ptr<ID3D12DescriptorHeap*[]> ppDescriptorHeapsM(new ID3D12DescriptorHeap*[ppDescriptorHeaps->Length]);
			for (int i = 0; i < ppDescriptorHeaps->Length; ++i)
			{
				ppDescriptorHeapsM[i] = ppDescriptorHeaps[i]->Wrapped;
			}
			pWrapped->SetDescriptorHeaps(ppDescriptorHeaps->Length, ppDescriptorHeapsM.get());
		}
		void SetGraphicsRoot32BitConstants(UINT RootParameterIndex, UINT Num32BitValuesToSet, cli::array<float> ^pSrcData, UINT DestOffsetIn32BitValues)
		{
			pin_ptr<float> pSrcDataM = &pSrcData[0];
			pWrapped->SetGraphicsRoot32BitConstants(RootParameterIndex, Num32BitValuesToSet, &pSrcDataM[0], DestOffsetIn32BitValues);
		}
		void SetGraphicsRootSignature(D3D12RootSignature ^pRootSignature)
		{
			pWrapped->SetGraphicsRootSignature(pRootSignature == nullptr ? nullptr : pRootSignature->Wrapped);
		}
		void SetPipelineState(D3D12PipelineState ^pPipelineState)
		{
			pWrapped->SetPipelineState(pPipelineState == nullptr ? nullptr : pPipelineState->Wrapped);
		}
		void ResourceBarrier(cli::array<D3D12ResourceBarrier> ^pBarriers)
		{
			std::unique_ptr<D3D12_RESOURCE_BARRIER[]> pBarriersM(new D3D12_RESOURCE_BARRIER[pBarriers->Length]);
			for (int i = 0; i < pBarriers->Length; ++i)
			{
				pBarriersM[i].Type = (D3D12_RESOURCE_BARRIER_TYPE)pBarriers[i].Type;
				pBarriersM[i].Flags = (D3D12_RESOURCE_BARRIER_FLAGS)pBarriers[i].Flags;
				pBarriersM[i].Transition.pResource = pBarriers[i].Transition.pResource->Wrapped;
				pBarriersM[i].Transition.Subresource = pBarriers[i].Transition.Subresource;
				pBarriersM[i].Transition.StateBefore = (D3D12_RESOURCE_STATES)pBarriers[i].Transition.StateBefore;
				pBarriersM[i].Transition.StateAfter = (D3D12_RESOURCE_STATES)pBarriers[i].Transition.StateAfter;
			}
			pWrapped->ResourceBarrier(pBarriers->Length, pBarriersM.get());
		}
	};
	#pragma endregion
	#pragma region - D3D12CommandQueue -
	public ref class D3D12CommandQueue : public COMWrapper<ID3D12CommandQueue>
	{
	public:
		D3D12CommandQueue(ID3D12CommandQueue *pObj) : COMWrapper(pObj)
		{
		}
		void ExecuteCommandLists(cli::array<D3D12GraphicsCommandList1^> ^ppCommandLists)
		{
			if (ppCommandLists->Length != 1) throw gcnew System::Exception("Unexpected command list count.");
			ID3D12CommandList *ppCommandLists2 = ppCommandLists[0]->Wrapped;
			pWrapped->ExecuteCommandLists(1, &ppCommandLists2);
		}
		void Signal(D3D12Fence ^pFence, UINT64 Value)
		{
			TRY_D3D(pWrapped->Signal(pFence->Wrapped, Value));
		}
		void Wait(D3D12Fence ^pFence, UINT64 Value)
		{
			TRY_D3D(pWrapped->Wait(pFence->Wrapped, Value));
		}
	};
	#pragma endregion
	#pragma region - D3D12Device -
	public ref class D3D12Device : public COMWrapper<ID3D12Device3>
	{
	public:
		D3D12Device(ID3D12Device3 *obj) : COMWrapper(obj)
		{
		}
		D3D12CommandAllocator^ CreateCommandAllocator(D3D12CommandListType type)
		{
			void *ppCommandAllocator = nullptr;
			TRY_D3D(pWrapped->CreateCommandAllocator((D3D12_COMMAND_LIST_TYPE)type, __uuidof(ID3D12CommandAllocator), &ppCommandAllocator));
			return gcnew D3D12CommandAllocator(reinterpret_cast<ID3D12CommandAllocator*>(ppCommandAllocator));
		}
		D3D12GraphicsCommandList1^ CreateCommandList(UINT nodeMask, D3D12CommandListType type, D3D12CommandAllocator ^pCommandAllocator, D3D12PipelineState ^pInitialState)
		{
			void *ppCommandList = nullptr;
			TRY_D3D(pWrapped->CreateCommandList(nodeMask, (D3D12_COMMAND_LIST_TYPE)type, pCommandAllocator->Wrapped, pInitialState->Wrapped, __uuidof(ID3D12GraphicsCommandList1), &ppCommandList));
			return gcnew D3D12GraphicsCommandList1(reinterpret_cast<ID3D12GraphicsCommandList1*>(ppCommandList));
		}
		D3D12CommandQueue^ CreateCommandQueue(D3D12CommandQueueDesc pDesc)
		{
			void *ppCommandQueue = nullptr;
			TRY_D3D(pWrapped->CreateCommandQueue((D3D12_COMMAND_QUEUE_DESC*)&pDesc, __uuidof(ID3D12CommandQueue), &ppCommandQueue));
			return gcnew D3D12CommandQueue(reinterpret_cast<ID3D12CommandQueue*>(ppCommandQueue));
		}
		D3D12Resource^ CreateCommittedResource(D3D12HeapProperties pHeapProperties, D3D12HeapFlags HeapFlags, D3D12ResourceDesc pDesc, D3D12ResourceStates InitialResourceState, System::Nullable<D3D12ClearValue> pOptimizedClearValue)
		{
			void *ppvResource = nullptr;
			TRY_D3D(pWrapped->CreateCommittedResource(reinterpret_cast<D3D12_HEAP_PROPERTIES*>(&pHeapProperties), (D3D12_HEAP_FLAGS)HeapFlags, reinterpret_cast<D3D12_RESOURCE_DESC*>(&pDesc), (D3D12_RESOURCE_STATES)InitialResourceState, pOptimizedClearValue.HasValue ? reinterpret_cast<D3D12_CLEAR_VALUE*>(&pOptimizedClearValue.Value) : nullptr, __uuidof(ID3D12Resource), &ppvResource));
			return gcnew D3D12Resource(reinterpret_cast<ID3D12Resource*>(ppvResource));
		}
		D3D12DescriptorHeap^ CreateDescriptorHeap(D3D12DescriptorHeapDesc pDescriptorHeapDesc)
		{
			void *ppvHeap = nullptr;
			TRY_D3D(pWrapped->CreateDescriptorHeap((D3D12_DESCRIPTOR_HEAP_DESC*)&pDescriptorHeapDesc, __uuidof(ID3D12DescriptorHeap), &ppvHeap));
			return gcnew D3D12DescriptorHeap(reinterpret_cast<ID3D12DescriptorHeap*>(ppvHeap));
		}
		D3D12Fence^ CreateFence(UINT64 InitialValue, D3D12FenceFlags Flags)
		{
			void *ppFence = nullptr;
			TRY_D3D(pWrapped->CreateFence(InitialValue, (D3D12_FENCE_FLAGS)Flags, __uuidof(ID3D12Fence), &ppFence));
			return gcnew D3D12Fence(reinterpret_cast<ID3D12Fence*>(ppFence));
		}
		D3D12PipelineState^ CreateGraphicsPipelineState(D3D12GraphicsPipelineStateDesc pDesc)
		{
			void *ppPipelineState = nullptr;
			D3D12_GRAPHICS_PIPELINE_STATE_DESC pDesc2 = { 0 };
			pDesc2.pRootSignature = pDesc.pRootSignature == nullptr ? nullptr : pDesc.pRootSignature->Wrapped;
			pin_ptr<byte> vs(&pDesc.VS[0]);
			pDesc2.VS.pShaderBytecode = vs;
			pDesc2.VS.BytecodeLength = pDesc.VS == nullptr ? 0 : pDesc.VS->Length;
			pin_ptr<byte> ps(&pDesc.PS[0]);
			pDesc2.PS.pShaderBytecode = ps;
			pDesc2.PS.BytecodeLength = pDesc.PS == nullptr ? 0 : pDesc.PS->Length;
			pDesc2.DS.BytecodeLength = pDesc.DS == nullptr ? 0 : pDesc.DS->Length;
			pDesc2.HS.BytecodeLength = pDesc.HS == nullptr ? 0 : pDesc.HS->Length;
			pDesc2.GS.BytecodeLength = pDesc.GS == nullptr ? 0 : pDesc.GS->Length;
			pDesc2.BlendState.AlphaToCoverageEnable = pDesc.BlendState.AlphaToCoverageEnable;
			pDesc2.BlendState.IndependentBlendEnable = pDesc.BlendState.IndependentBlendEnable;
			pDesc2.BlendState.RenderTarget[0].BlendEnable = pDesc.BlendState.RenderTarget0.BlendEnable;
			pDesc2.BlendState.RenderTarget[0].LogicOpEnable = pDesc.BlendState.RenderTarget0.LogicOpEnable;
			pDesc2.BlendState.RenderTarget[0].SrcBlend = (D3D12_BLEND)pDesc.BlendState.RenderTarget0.SrcBlend;
			pDesc2.BlendState.RenderTarget[0].DestBlend = (D3D12_BLEND)pDesc.BlendState.RenderTarget0.DestBlend;
			pDesc2.BlendState.RenderTarget[0].BlendOp = (D3D12_BLEND_OP)pDesc.BlendState.RenderTarget0.BlendOp;
			pDesc2.BlendState.RenderTarget[0].SrcBlendAlpha = (D3D12_BLEND)pDesc.BlendState.RenderTarget0.SrcBlendAlpha;
			pDesc2.BlendState.RenderTarget[0].DestBlendAlpha = (D3D12_BLEND)pDesc.BlendState.RenderTarget0.DestBlendAlpha;
			pDesc2.BlendState.RenderTarget[0].BlendOpAlpha = (D3D12_BLEND_OP)pDesc.BlendState.RenderTarget0.BlendOpAlpha;
			pDesc2.BlendState.RenderTarget[0].LogicOp = (D3D12_LOGIC_OP)pDesc.BlendState.RenderTarget0.LogicOp;
			pDesc2.BlendState.RenderTarget[0].RenderTargetWriteMask = pDesc.BlendState.RenderTarget0.RenderTargetWriteMask;
			pDesc2.SampleMask = pDesc.SampleMask;
			pDesc2.RasterizerState.CullMode = (D3D12_CULL_MODE)pDesc.RasterizerState.CullMode;
			pDesc2.RasterizerState.FillMode = (D3D12_FILL_MODE)pDesc.RasterizerState.FillMode;
			pDesc2.SampleDesc.Count = pDesc.SampleDesc.Count;
			pDesc2.SampleDesc.Quality = pDesc.SampleDesc.Quality;
			D3D12_INPUT_ELEMENT_DESC inputlayout;
			inputlayout.SemanticName = "POSITION";
			inputlayout.SemanticIndex = pDesc.InputLayout.pInputElementDescs[0].SemanticIndex;
			inputlayout.Format = (DXGI_FORMAT)pDesc.InputLayout.pInputElementDescs[0].Format;
			inputlayout.InputSlot = pDesc.InputLayout.pInputElementDescs[0].InputSlot;
			inputlayout.AlignedByteOffset = pDesc.InputLayout.pInputElementDescs[0].AlignedByteOffset;
			inputlayout.InputSlotClass = (D3D12_INPUT_CLASSIFICATION)pDesc.InputLayout.pInputElementDescs[0].InputSlotClass;
			inputlayout.InstanceDataStepRate = pDesc.InputLayout.pInputElementDescs[0].InstanceDataStepRate;
			pDesc2.InputLayout.pInputElementDescs = &inputlayout;
			pDesc2.InputLayout.NumElements = pDesc.InputLayout.pInputElementDescs->Length;
			pDesc2.PrimitiveTopologyType = (D3D12_PRIMITIVE_TOPOLOGY_TYPE)pDesc.PrimitiveTopologyType;
			pDesc2.NumRenderTargets = pDesc.NumRenderTargets;
			pDesc2.RTVFormats[0] = (DXGI_FORMAT)pDesc.RTVFormats0;
			TRY_D3D(pWrapped->CreateGraphicsPipelineState(&pDesc2, __uuidof(ID3D12PipelineState), &ppPipelineState));
			return gcnew D3D12PipelineState(reinterpret_cast<ID3D12PipelineState*>(ppPipelineState));
		}
		void CreateRenderTargetView(D3D12Resource ^pResource, D3D12RenderTargetViewDesc pDesc, D3D12CPUDescriptorHandle DestDescriptor)
		{
			D3D12_RENDER_TARGET_VIEW_DESC pDesc2;
			pDesc2.Format = (DXGI_FORMAT)pDesc.Format;
			pDesc2.ViewDimension = (D3D12_RTV_DIMENSION)pDesc.ViewDimension;
			pDesc2.Texture2D.MipSlice = pDesc.Texture2D.MipSlice;
			pDesc2.Texture2D.PlaneSlice = pDesc.Texture2D.PlaneSlice;
			D3D12_CPU_DESCRIPTOR_HANDLE DestDescriptor2;
			DestDescriptor2.ptr = (SIZE_T)DestDescriptor.ptr.ToPointer();
			pWrapped->CreateRenderTargetView(pResource->Wrapped, &pDesc2, DestDescriptor2);
		}
		D3D12RootSignature^ CreateRootSignature(UINT nodeMask, cli::array<byte> ^pBlobWithRootSignature)
		{
			void* ppvRootSignature = nullptr;
			pin_ptr<byte> pBlobWithRootSignatureM = &pBlobWithRootSignature[0];
			TRY_D3D(pWrapped->CreateRootSignature(nodeMask, pBlobWithRootSignatureM, pBlobWithRootSignature->Length, __uuidof(ID3D12RootSignature), &ppvRootSignature));
			return gcnew D3D12RootSignature(reinterpret_cast<ID3D12RootSignature*>(ppvRootSignature));
		}
	};
	#pragma endregion
	#pragma region - D3D12Debug -
	public ref class D3D12Debug : public COMWrapper<ID3D12Debug>
	{
	public:
		D3D12Debug(ID3D12Debug *obj) : COMWrapper(obj)
		{
		}
		void EnableDebugLayer()
		{
			pWrapped->EnableDebugLayer();
		}
	};
	#pragma endregion
	#pragma region - Direct3D12 Global Functions -
	public ref class Direct3D12
	{
	public:
		static D3D12Debug^ D3D12GetDebugInterface()
		{
			void *ppvDebug = nullptr;
			TRY_D3D(::D3D12GetDebugInterface(__uuidof(ID3D12Debug), &ppvDebug));
			return gcnew D3D12Debug(reinterpret_cast<ID3D12Debug*>(ppvDebug));
		}
		static D3D12Device^ D3D12CreateDevice()
		{
			void *ppDevice = nullptr;
			TRY_D3D(::D3D12CreateDevice(nullptr, D3D_FEATURE_LEVEL_12_1, _uuidof(ID3D12Device3), &ppDevice));
			return gcnew D3D12Device(reinterpret_cast<ID3D12Device3*>(ppDevice));
		}
	};
	#pragma endregion
}