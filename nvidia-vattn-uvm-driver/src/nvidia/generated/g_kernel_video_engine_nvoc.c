#define NVOC_KERNEL_VIDEO_ENGINE_H_PRIVATE_ACCESS_ALLOWED
#include "nvoc/runtime.h"
#include "nvoc/rtti.h"
#include "nvtypes.h"
#include "nvport/nvport.h"
#include "nvport/inline/util_valist.h"
#include "utils/nvassert.h"
#include "g_kernel_video_engine_nvoc.h"

#ifdef DEBUG
char __nvoc_class_id_uniqueness_check_0x9e2f3e = 1;
#endif

extern const struct NVOC_CLASS_DEF __nvoc_class_def_KernelVideoEngine;

extern const struct NVOC_CLASS_DEF __nvoc_class_def_Object;

void __nvoc_init_KernelVideoEngine(KernelVideoEngine*);
void __nvoc_init_funcTable_KernelVideoEngine(KernelVideoEngine*);
NV_STATUS __nvoc_ctor_KernelVideoEngine(KernelVideoEngine*, struct OBJGPU * arg_pGpu, ENGDESCRIPTOR arg_physEngDesc);
void __nvoc_init_dataField_KernelVideoEngine(KernelVideoEngine*);
void __nvoc_dtor_KernelVideoEngine(KernelVideoEngine*);
extern const struct NVOC_EXPORT_INFO __nvoc_export_info_KernelVideoEngine;

static const struct NVOC_RTTI __nvoc_rtti_KernelVideoEngine_KernelVideoEngine = {
    /*pClassDef=*/          &__nvoc_class_def_KernelVideoEngine,
    /*dtor=*/               (NVOC_DYNAMIC_DTOR) &__nvoc_dtor_KernelVideoEngine,
    /*offset=*/             0,
};

static const struct NVOC_RTTI __nvoc_rtti_KernelVideoEngine_Object = {
    /*pClassDef=*/          &__nvoc_class_def_Object,
    /*dtor=*/               &__nvoc_destructFromBase,
    /*offset=*/             NV_OFFSETOF(KernelVideoEngine, __nvoc_base_Object),
};

static const struct NVOC_CASTINFO __nvoc_castinfo_KernelVideoEngine = {
    /*numRelatives=*/       2,
    /*relatives=*/ {
        &__nvoc_rtti_KernelVideoEngine_KernelVideoEngine,
        &__nvoc_rtti_KernelVideoEngine_Object,
    },
};

const struct NVOC_CLASS_DEF __nvoc_class_def_KernelVideoEngine = 
{
    /*classInfo=*/ {
        /*size=*/               sizeof(KernelVideoEngine),
        /*classId=*/            classId(KernelVideoEngine),
        /*providerId=*/         &__nvoc_rtti_provider,
#if NV_PRINTF_STRINGS_ALLOWED
        /*name=*/               "KernelVideoEngine",
#endif
    },
    /*objCreatefn=*/        (NVOC_DYNAMIC_OBJ_CREATE) &__nvoc_objCreateDynamic_KernelVideoEngine,
    /*pCastInfo=*/          &__nvoc_castinfo_KernelVideoEngine,
    /*pExportInfo=*/        &__nvoc_export_info_KernelVideoEngine
};

const struct NVOC_EXPORT_INFO __nvoc_export_info_KernelVideoEngine = 
{
    /*numEntries=*/     0,
    /*pExportEntries=*/  0
};

void __nvoc_dtor_Object(Object*);
void __nvoc_dtor_KernelVideoEngine(KernelVideoEngine *pThis) {
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_dataField_KernelVideoEngine(KernelVideoEngine *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

NV_STATUS __nvoc_ctor_Object(Object* );
NV_STATUS __nvoc_ctor_KernelVideoEngine(KernelVideoEngine *pThis, struct OBJGPU * arg_pGpu, ENGDESCRIPTOR arg_physEngDesc) {
    NV_STATUS status = NV_OK;
    status = __nvoc_ctor_Object(&pThis->__nvoc_base_Object);
    if (status != NV_OK) goto __nvoc_ctor_KernelVideoEngine_fail_Object;
    __nvoc_init_dataField_KernelVideoEngine(pThis);

    status = __nvoc_kvidengConstruct(pThis, arg_pGpu, arg_physEngDesc);
    if (status != NV_OK) goto __nvoc_ctor_KernelVideoEngine_fail__init;
    goto __nvoc_ctor_KernelVideoEngine_exit; // Success

__nvoc_ctor_KernelVideoEngine_fail__init:
    __nvoc_dtor_Object(&pThis->__nvoc_base_Object);
__nvoc_ctor_KernelVideoEngine_fail_Object:
__nvoc_ctor_KernelVideoEngine_exit:

    return status;
}

static void __nvoc_init_funcTable_KernelVideoEngine_1(KernelVideoEngine *pThis) {
    PORT_UNREFERENCED_VARIABLE(pThis);
}

void __nvoc_init_funcTable_KernelVideoEngine(KernelVideoEngine *pThis) {
    __nvoc_init_funcTable_KernelVideoEngine_1(pThis);
}

void __nvoc_init_Object(Object*);
void __nvoc_init_KernelVideoEngine(KernelVideoEngine *pThis) {
    pThis->__nvoc_pbase_KernelVideoEngine = pThis;
    pThis->__nvoc_pbase_Object = &pThis->__nvoc_base_Object;
    __nvoc_init_Object(&pThis->__nvoc_base_Object);
    __nvoc_init_funcTable_KernelVideoEngine(pThis);
}

NV_STATUS __nvoc_objCreate_KernelVideoEngine(KernelVideoEngine **ppThis, Dynamic *pParent, NvU32 createFlags, struct OBJGPU * arg_pGpu, ENGDESCRIPTOR arg_physEngDesc) {
    NV_STATUS status;
    Object *pParentObj;
    KernelVideoEngine *pThis;

    status = __nvoc_handleObjCreateMemAlloc(createFlags, sizeof(KernelVideoEngine), (void**)&pThis, (void**)ppThis);
    if (status != NV_OK)
        return status;

    portMemSet(pThis, 0, sizeof(KernelVideoEngine));

    __nvoc_initRtti(staticCast(pThis, Dynamic), &__nvoc_class_def_KernelVideoEngine);

    pThis->__nvoc_base_Object.createFlags = createFlags;

    if (pParent != NULL && !(createFlags & NVOC_OBJ_CREATE_FLAGS_PARENT_HALSPEC_ONLY))
    {
        pParentObj = dynamicCast(pParent, Object);
        objAddChild(pParentObj, &pThis->__nvoc_base_Object);
    }
    else
    {
        pThis->__nvoc_base_Object.pParent = NULL;
    }

    __nvoc_init_KernelVideoEngine(pThis);
    status = __nvoc_ctor_KernelVideoEngine(pThis, arg_pGpu, arg_physEngDesc);
    if (status != NV_OK) goto __nvoc_objCreate_KernelVideoEngine_cleanup;

    *ppThis = pThis;

    return NV_OK;

__nvoc_objCreate_KernelVideoEngine_cleanup:
    // do not call destructors here since the constructor already called them
    if (createFlags & NVOC_OBJ_CREATE_FLAGS_IN_PLACE_CONSTRUCT)
        portMemSet(pThis, 0, sizeof(KernelVideoEngine));
    else
        portMemFree(pThis);

    // coverity[leaked_storage:FALSE]
    return status;
}

NV_STATUS __nvoc_objCreateDynamic_KernelVideoEngine(KernelVideoEngine **ppThis, Dynamic *pParent, NvU32 createFlags, va_list args) {
    NV_STATUS status;
    struct OBJGPU * arg_pGpu = va_arg(args, struct OBJGPU *);
    ENGDESCRIPTOR arg_physEngDesc = va_arg(args, ENGDESCRIPTOR);

    status = __nvoc_objCreate_KernelVideoEngine(ppThis, pParent, createFlags, arg_pGpu, arg_physEngDesc);

    return status;
}

