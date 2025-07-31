//! File/code adapted from https://github.com/jamjamjon/usls

use anyhow::Result;
use half::{bf16, f16};
use ndarray::{Array, IxDyn};
use prost::Message;
use std::collections::HashSet;
use std::path::PathBuf;
use std::str::FromStr;
use aksr::Builder;
use bvr_common::InferenceDevice;
use ort::{
    execution_providers::{ExecutionProvider,
                          CPUExecutionProvider,
                          CUDAExecutionProvider,
                          TensorRTExecutionProvider,
                          CoreMLExecutionProvider},
    session::builder::SessionBuilder,
    session::{Session,SessionInputValue},
    tensor::TensorElementType,
    value::Value
};
use ort::execution_providers::cuda::CUDAAttentionBackend;
use ort::value::DynValue;
use ort_sys::OrtCUDAProviderOptionsV2;
use crate::data::{MinOptMax, ConfigOrt, TimeCalc, Xs};
use crate::detection_runners::ort_detector::onnx;
use crate::data::CROSS_MARK;
use crate::detection_runners::image_ops::make_divisible;
use crate::detection_runners::input_wrapper::X;
use crate::utils::human_bytes;

/// A struct for input composed of the i-th input, the ii-th dimension, and the value.
#[derive(Clone, Debug, Default)]
pub struct Iiix {
    pub i: usize,
    pub ii: usize,
    pub x: MinOptMax,
}

impl From<(usize, usize, MinOptMax)> for Iiix {
    fn from((i, ii, x): (usize, usize, MinOptMax)) -> Self {
        Self { i, ii, x }
    }
}

/// A struct for tensor attrs composed of the names, the dtypes, and the dimensions.
/// ONNX Runtime tensor attributes containing names, data types, and dimensions.
#[derive(Builder, Debug, Clone, Default)]
/// ONNX Runtime tensor attributes containing metadata.
pub struct OrtTensorAttr {
    /// Tensor names.
    pub names: Vec<String>,
    /// Tensor data types.
    pub dtypes: Vec<TensorElementType>,
    /// Tensor dimensions for each tensor.
    pub dimss: Vec<Vec<usize>>,
}

/// ONNXRuntime Backend
#[derive(Debug)]
pub struct OrtEngine {
    //name: String,
    session: Session,
    device: InferenceDevice,
    inputs_min_opt_max: Vec<Vec<MinOptMax>>,
    inputs_attrs: OrtTensorAttr,
    outputs_attrs: OrtTensorAttr,
    model_width: u32,
    model_height: u32,
    profile: bool,
    model_proto: onnx::ModelProto,
    params: usize,
    wb_mem: usize,
    pub infer_time: TimeCalc,
}

#[allow(dead_code)]
impl OrtEngine {
    pub fn new(config: &ConfigOrt) -> Result<Self> {
        // onnx graph
        let model_proto = Self::load_onnx(&config.onnx_path)?;

        let graph = match &model_proto.graph {
            Some(graph) => graph,
            None => anyhow::bail!("No graph found in this proto. Failed to parse ONNX model."),
        };
        // model params & mems
        let byte_alignment = 16; // 16 for simd; 8 for most
        let mut params: usize = 0;
        let mut wbmems: usize = 0;
        let mut initializer_names: HashSet<&str> = HashSet::new();
        for tensor_proto in graph.initializer.iter() {
            initializer_names.insert(&tensor_proto.name);
            let param = tensor_proto.dims.iter().product::<i64>() as usize;
            params += param;

            // mems
            let param = make_divisible(param, byte_alignment);
            let n = Self::nbytes_from_onnx_dtype_id(tensor_proto.data_type as usize);
            let wbmem = param * n;
            wbmems += wbmem;
        }

        // inputs & outputs
        let inputs_attrs = Self::io_from_onnx_value_info(&initializer_names, &graph.input)?;
        println!("INPUT ATTRIBUTES: {:?}", inputs_attrs);
        let outputs_attrs = Self::io_from_onnx_value_info(&initializer_names, &graph.output)?;
        let inputs_minoptmax =
            Self::build_inputs_minoptmax(&inputs_attrs, &config.iiixs, config.batch_size)?;

        // build
        let ort_init = ort::init_from(&config.ort_lib_path);
        //ort_init.commit().expect("Failed to commit ONNX dependencies");

        match ort_init.commit() {
            Ok(_) => {},
            Err(e) => {
                println!("ORT commit failed! Error: {:?}", e);
                return Err(anyhow::anyhow!("Failed to commit ORT: {:?}", e));
            }
        };

        let mut builder = Session::builder()?;

        let mut device = config.device.to_owned();
        match device {
            InferenceDevice::TensorRT(device_id) => {
                Self::build_trt(
                    &inputs_attrs.names,
                    &inputs_minoptmax,
                    &mut builder,
                    device_id,
                    config.trt_int8_enable,
                    config.trt_fp16_enable,
                    config.trt_engine_cache_enable,
                )?;
            }
            InferenceDevice::CUDA(device_id) => {
                Self::build_cuda(&mut builder, device_id).unwrap_or_else(|err| {
                    log::warn!("{err}, Using cpu");
                    device = InferenceDevice::CPU;
                })
            }
            InferenceDevice::CoreML(_) => Self::build_coreml(&mut builder).unwrap_or_else(|err| {
                log::warn!("{err}, Using cpu");
                device = InferenceDevice::CPU;
            }),
            InferenceDevice::CPU => {
                Self::build_cpu(&mut builder)?;
            }
        }

        let session = builder
            .with_optimization_level(ort::session::builder::GraphOptimizationLevel::Level3)?
            .commit_from_file(&config.onnx_path)?;

        // summary
        log::info!(
            "Backend: ONNXRuntime | Opset: {} | Device: {:?} | Params: {}",
            model_proto.opset_import[0].version,
            device,
            human_bytes(params as f64),
        );

        Ok(Self {
            //name: config.onnx_path.to_owned(),
            session,
            device,
            inputs_min_opt_max: inputs_minoptmax,
            inputs_attrs,
            outputs_attrs,
            model_width: config.model_width,
            model_height: config.model_height,
            profile: config.profile,
            model_proto,
            params,
            wb_mem: wbmems,
            infer_time: TimeCalc::default(),
        })
    }

    fn build_trt(
        names: &[String],
        inputs_minoptmax: &[Vec<MinOptMax>],
        builder: &mut SessionBuilder,
        device_id: usize,
        int8_enable: bool,
        fp16_enable: bool,
        engine_cache_enable: bool,
    ) -> Result<()> {
        // auto generate shapes
        let mut spec_min = String::new();
        let mut spec_opt = String::new();
        let mut spec_max = String::new();
        for (i, name) in names.iter().enumerate() {
            if i != 0 {
                spec_min.push(',');
                spec_opt.push(',');
                spec_max.push(',');
            }
            let mut s_min = format!("{}:", name);
            let mut s_opt = format!("{}:", name);
            let mut s_max = format!("{}:", name);
            for d in inputs_minoptmax[i].iter() {
                let min_ = &format!("{}x", d.min());
                let opt_ = &format!("{}x", d.opt());
                let max_ = &format!("{}x", d.max());
                s_min += min_;
                s_opt += opt_;
                s_max += max_;
            }
            s_min.pop();
            s_opt.pop();
            s_max.pop();
            spec_min += &s_min;
            spec_opt += &s_opt;
            spec_max += &s_max;
        }
        let trt = TensorRTExecutionProvider::default()
            .with_device_id(device_id as i32)
            .with_int8(int8_enable)
            .with_int8_calibration_table_name("/home/chris/Documents/BvrWebPup/trt_int8_calibration_table")
            .with_fp16(fp16_enable)
            .with_engine_cache(engine_cache_enable)
            .with_engine_cache_path("trt-cache")
            .with_timing_cache(false)
            .with_profile_min_shapes(spec_min)
            .with_profile_opt_shapes(spec_opt)
            .with_profile_max_shapes(spec_max);
        if trt.is_available()? {
            match trt.register(builder) {
                Ok(_) => { }
                Err(err) => { anyhow::bail!("{CROSS_MARK} TensorRT initialization failed: {:?}", err) }
            }
            log::info!("🐢 Initial model serialization with TensorRT may takes some time...\n");
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} TensorRT execution provider not available")
        }
    }

    fn build_cuda(builder: &mut SessionBuilder, device_id: usize) -> Result<()> {
        let ep = CUDAExecutionProvider::default()
            .with_attention_backend(CUDAAttentionBackend::LEAN_ATTENTION)
            .with_device_id(device_id as i32);
        if ep.is_available()? {
            match ep.register(builder) {
                Ok(_) => { }
                Err(err) => { anyhow::bail!("{CROSS_MARK} CUDA initialization failed: {:?}", err) }
            }
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} CUDA execution provider not available")
        }
    }

    fn build_coreml(builder: &mut SessionBuilder) -> Result<()> {
        let ep = CoreMLExecutionProvider::default()
            .with_subgraphs(false);
            //.with_ane_only();
        if ep.is_available()? {
            match ep.register(builder) {
                Ok(_) => { }
                Err(err) => { anyhow::bail!("{CROSS_MARK} CoreML initialization failed: {:?}", err) }
            }
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} CoreML execution provider not available")
        }
    }

    fn build_cpu(builder: &mut SessionBuilder) -> Result<()> {
        let ep = CPUExecutionProvider::default();
        if ep.is_available()? {
            match ep.register(builder) {
                Ok(_) => { }
                Err(err) => { anyhow::bail!("{CROSS_MARK} CPU initialization failed: {:?}", err) }
            }
            Ok(())
        } else {
            anyhow::bail!("{CROSS_MARK} CPU execution provider not available")
        }
    }

    fn tensor_preprocess(x: &X, dtype: &TensorElementType) -> Result<DynValue> {
        let x = match dtype {
            TensorElementType::Float32 | TensorElementType::Float64 => {
                Value::from_array(x.0.clone())?.into_dyn()
            }
            TensorElementType::Float16 => Value::from_array(x.mapv(f16::from_f32))?.into_dyn(),
            TensorElementType::Bfloat16 => Value::from_array(x.mapv(bf16::from_f32))?.into_dyn(),
            TensorElementType::Int8 => Value::from_array(x.mapv(|x_| x_ as i8))?.into_dyn(),
            TensorElementType::Int16 => Value::from_array(x.mapv(|x_| x_ as i16))?.into_dyn(),
            TensorElementType::Int32 => Value::from_array(x.mapv(|x_| x_ as i32))?.into_dyn(),
            TensorElementType::Int64 => Value::from_array(x.mapv(|x_| x_ as i64))?.into_dyn(),
            TensorElementType::Uint8 => Value::from_array(x.mapv(|x_| x_ as u8))?.into_dyn(),
            TensorElementType::Uint16 => Value::from_array(x.mapv(|x_| x_ as u16))?.into_dyn(),
            TensorElementType::Uint32 => Value::from_array(x.mapv(|x_| x_ as u32))?.into_dyn(),
            TensorElementType::Uint64 => Value::from_array(x.mapv(|x_| x_ as u64))?.into_dyn(),
            TensorElementType::Bool => Value::from_array(x.mapv(|x_| x_ != 0.))?.into_dyn(),
            _ => unimplemented!(),
        };
        Ok(x)
    }

    pub fn engine_run(&mut self, xs: Xs) -> Result<Xs> {
        //let mut ys = xs.derive();
        let mut ys = Xs::new();

        // alignment
        let mut xs_ = Vec::new();
        for (dtype, x) in self.inputs_attrs.dtypes.iter().zip(xs.into_iter()) {
            xs_.push(Into::<SessionInputValue<'_>>::into(Self::tensor_preprocess(
                x, dtype,
            )?));
        }

        // run
        let outputs_res = self.session.run(&xs_[..]);

        let outputs = match outputs_res {
            Ok(out) => out,
            Err(err) => { panic!("{:?}", err) }
        };

        // extract
        for (dtype, name) in self.outputs_attrs.dtypes.iter().zip(self.outputs_attrs.names.iter()) {
            let y = Self::tensor_postprocess(&outputs[name.as_str()], dtype)?;
            ys.push_kv(name.as_str(), X::from(y))?;
        }

        Ok(ys)
    }

/*    pub fn run(&mut self, xs: Xs) -> Result<Xs> {
        // inputs dtype alignment
        let mut xs_ = Vec::new();
        let t_pre = std::time::Instant::now();
        for (idtype, x) in self.inputs_attrs.dtypes.iter().zip(xs.into_iter()) {
            let x_ = match &idtype {
                TensorElementType::Float32 => Value::from_array(x.view())?.into_dyn(),
                TensorElementType::Float16 => {
                    Value::from_array(x.mapv(f16::from_f32).view())?.into_dyn()
                }
                TensorElementType::Int32 => {
                    Value::from_array(x.mapv(|x_| x_ as i32).view())?.into_dyn()
                }
                TensorElementType::Int64 => {
                    Value::from_array(x.mapv(|x_| x_ as i64).view())?.into_dyn()
                }
                TensorElementType::Uint8 => {
                    Value::from_array(x.mapv(|x_| x_ as u8).view())?.into_dyn()
                }
                TensorElementType::Int8 => {
                    Value::from_array(x.mapv(|x_| x_ as i8).view())?.into_dyn()
                }
                TensorElementType::Bool => {
                    Value::from_array(x.mapv(|x_| x_ != 0.).view())?.into_dyn()
                }
                _ => todo!(),
            };
            xs_.push(Into::<SessionInputValue<'_>>::into(x_));
        }
        let t_pre = t_pre.elapsed();
        self.infer_time.add_or_push(0, t_pre);

        // inference
        let t_run = std::time::Instant::now();
        let outputs = self.session.run(&xs_[..])?;

        let t_run = t_run.elapsed();
        self.infer_time.add_or_push(1, t_run);

        // output
        let mut ys = Xs::new();
        let t_post = std::time::Instant::now();
        for (dtype, name) in self
            .outputs_attrs
            .dtypes
            .iter()
            .zip(self.outputs_attrs.names.iter())
        {
            let y = &outputs[name.as_str()];

            let y_ = match &dtype {
                TensorElementType::Float32 => match y.try_extract_tensor::<f32>() {
                    Err(err) => {
                        log::error!("Error: {:?}. Output name: {:?}", err, name);
                        Array::zeros(0).into_dyn()
                    }
                    Ok(x) => x.view().into_owned(),
                },
                TensorElementType::Float16 => match y.try_extract_tensor::<f16>() {
                    Err(err) => {
                        log::error!("Error: {:?}. Output name: {:?}", err, name);
                        Array::zeros(0).into_dyn()
                    }
                    Ok(x) => x.view().mapv(f16::to_f32).into_owned(),
                },
                TensorElementType::Int64 => match y.try_extract_tensor::<i64>() {
                    Err(err) => {
                        log::error!("Error: {:?}. Output name: {:?}", err, name);
                        Array::zeros(0).into_dyn()
                    }
                    Ok(x) => x.view().to_owned().mapv(|x| x as f32).into_owned(),
                },
                _ => todo!(),
            };

            ys.push_kv(name.as_str(), X::from(y_))?;
        }
        let t_post = t_post.elapsed();
        self.infer_time.add_or_push(2, t_post);

        if self.profile {
            let len = 10usize;
            let n = 4usize;
            log::info!("[Profile] {:>len$.n$?} ({:>len$.n$?} avg) [alignment: {:>len$.n$?} ({:>len$.n$?} avg) | inference: {:>len$.n$?} ({:>len$.n$?} avg) | to_f32: {:>len$.n$?} ({:>len$.n$?} avg)]",
                t_pre + t_run + t_post,
                self.infer_time.avg(),
                t_pre,
                self.infer_time.avg_i(0),
                t_run,
                self.infer_time.avg_i(1),
                t_post,
                self.infer_time.avg_i(2),
            );
        }
        Ok(ys)
    }*/

    fn tensor_postprocess(x: &DynValue, dtype: &TensorElementType) -> Result<Array<f32, IxDyn>> {
        fn _extract_and_convert<T>(x: &DynValue, map_fn: impl Fn(T) -> f32) -> Array<f32, IxDyn>
        where
            T: Clone + 'static + ort::tensor::PrimitiveTensorElementType,
        {
            // for version 2.0.0-rc.10 of ORT
            match x.try_extract_array::<T>() {
                Err(_err) => {
                    //debug!("Failed to extract from ort outputs: {:?}. A default value has been generated.", err);
                    Array::zeros(0).into_dyn()
                }
                Ok(x) => x.view().mapv(map_fn).into_owned(),
            }
/*            match x.try_extract_tensor::<T>() {
                Err(err) => {
                    // Optional: log or debug here
                    Array::zeros(IxDyn(&[]))
                }
                Ok(tensor) => tensor.view().mapv(map_fn).into_owned(),
            }*/
        }
        let x = match dtype {
            TensorElementType::Float32 => _extract_and_convert::<f32>(x, |x| x),
            TensorElementType::Float16 => _extract_and_convert::<f16>(x, f16::to_f32),
            TensorElementType::Bfloat16 => _extract_and_convert::<bf16>(x, bf16::to_f32),
            TensorElementType::Float64 => _extract_and_convert::<f64>(x, |x| x as f32),
            TensorElementType::Int64 => _extract_and_convert::<i64>(x, |x| x as f32),
            TensorElementType::Int32 => _extract_and_convert::<i32>(x, |x| x as f32),
            TensorElementType::Int16 => _extract_and_convert::<i16>(x, |x| x as f32),
            TensorElementType::Int8 => _extract_and_convert::<i8>(x, |x| x as f32),
            TensorElementType::Uint64 => _extract_and_convert::<u64>(x, |x| x as f32),
            TensorElementType::Uint32 => _extract_and_convert::<u32>(x, |x| x as f32),
            TensorElementType::Uint16 => _extract_and_convert::<u16>(x, |x| x as f32),
            TensorElementType::Uint8 => _extract_and_convert::<u8>(x, |x| x as f32),
            TensorElementType::Bool => _extract_and_convert::<bool>(x, |x| x as u8 as f32),
            _ => return Err(anyhow::anyhow!("Unsupported ort tensor type: {:?}", dtype)),
        };

        Ok(x)
    }

    fn build_inputs_minoptmax(
        inputs_attrs: &OrtTensorAttr,
        iiixs: &[Iiix],
        batch_size: usize,
    ) -> Result<Vec<Vec<MinOptMax>>> {
        // init
        let mut ys: Vec<Vec<MinOptMax>> = inputs_attrs
            .dimss
            .iter()
            .map(|dims| dims.iter().map(|&x| MinOptMax::from(x)).collect())
            .collect();

        // update from customized
        for iiix in iiixs.iter() {
            if let Some(x) = inputs_attrs
                .dimss
                .get(iiix.i)
                .and_then(|dims| dims.get(iiix.ii))
            {
                // dynamic
                if *x == 0 {
                    ys[iiix.i][iiix.ii] = iiix.x.clone();
                }
            } else {
                anyhow::bail!(
                    "Cannot retrieve the {}-th dimension of the {}-th input.",
                    iiix.ii,
                    iiix.i,
                );
            }
        }

        // deal with the dynamic axis
        ys.iter_mut().enumerate().for_each(|(i, xs)| {
            xs.iter_mut().enumerate().for_each(|(ii, x)| {
                if x.is_dyn() {
                    let n = if ii == 0 { batch_size } else { 1 };
                    let y = MinOptMax::from(n);
                    log::warn!("Using dynamic shapes in inputs without specifying it: the {}-th input, the {}-th dimension. \
                        Using {:?} by default. You should make it clear when using TensorRT.",
                        i + 1, ii + 1, y
                    );
                    *x = y;
                }
            });
        });

        Ok(ys)
    }

    #[allow(dead_code)]
    fn nbytes_from_onnx_dtype_id(x: usize) -> usize {
        match x {
            7 | 11 | 13 => 8,     // i64, f64, u64
            1 | 6 | 12 => 4,      // f32, i32, u32
            10 | 16 | 5 | 4 => 2, // f16, bf16, i16, u16
            2 | 3 | 9 => 1,       // u8, i8, bool
            8 => 4,               // string(1~4)
            _ => todo!(),
        }
    }

    #[allow(dead_code)]
    fn nbytes_from_onnx_dtype(x: &TensorElementType) -> usize {
        match x {
            TensorElementType::Float64
            | TensorElementType::Uint64
            | TensorElementType::Int64 => 8, // i64, f64, u64
            TensorElementType::Float32
            | TensorElementType::Uint32
            | TensorElementType::Int32
            | TensorElementType::String => 4, // f32, i32, u32, string(1~4)
            TensorElementType::Float16
            | TensorElementType::Bfloat16
            | TensorElementType::Int16
            | TensorElementType::Uint16 => 2, // f16, bf16, i16, u16
            TensorElementType::Uint8
            | TensorElementType::Int8
            | TensorElementType::Bool => 1, // u8, i8, bool
            _ => { todo!() }
        }
    }

    #[allow(dead_code)]
    fn ort_dtype_from_onnx_dtype_id(value: i32) -> Option<TensorElementType> {
        match value {
            0 => None,
            1 => Some(TensorElementType::Float32),
            2 => Some(TensorElementType::Uint8),
            3 => Some(TensorElementType::Int8),
            4 => Some(TensorElementType::Uint16),
            5 => Some(TensorElementType::Int16),
            6 => Some(TensorElementType::Int32),
            7 => Some(TensorElementType::Int64),
            8 => Some(TensorElementType::String),
            9 => Some(TensorElementType::Bool),
            10 => Some(TensorElementType::Float16),
            11 => Some(TensorElementType::Float64),
            12 => Some(TensorElementType::Uint32),
            13 => Some(TensorElementType::Uint64),
            14 => None, // COMPLEX64
            15 => None, // COMPLEX128
            16 => Some(TensorElementType::Bfloat16),
            _ => None,
        }
    }

    fn io_from_onnx_value_info(
        initializer_names: &HashSet<&str>,
        value_info: &[onnx::ValueInfoProto],
    ) -> Result<OrtTensorAttr> {
        let mut dimss: Vec<Vec<usize>> = Vec::new();
        let mut dtypes: Vec<TensorElementType> = Vec::new();
        let mut names: Vec<String> = Vec::new();
        for v in value_info.iter() {
            if initializer_names.contains(v.name.as_str()) {
                continue;
            }
            names.push(v.name.to_string());
            let dtype = match &v.r#type {
                Some(dtype) => dtype,
                None => continue,
            };
            let dtype = match &dtype.value {
                Some(dtype) => dtype,
                None => continue,
            };
            let tensor = match dtype {
                onnx::type_proto::Value::TensorType(tensor) => tensor,
                _ => continue,
            };
            let tensor_type = tensor.elem_type;
            let tensor_type = match Self::ort_dtype_from_onnx_dtype_id(tensor_type) {
                Some(dtype) => dtype,
                None => continue,
            };
            dtypes.push(tensor_type);

            let shapes = match &tensor.shape {
                Some(shapes) => shapes,
                None => continue,
            };
            let mut shape_: Vec<usize> = Vec::new();
            for shape in shapes.dim.iter() {
                match &shape.value {
                    None => continue,
                    Some(value) => match value {
                        onnx::tensor_shape_proto::dimension::Value::DimValue(x) => {
                            shape_.push(*x as _);
                        }
                        onnx::tensor_shape_proto::dimension::Value::DimParam(_) => {
                            shape_.push(0);
                        }
                    },
                }
            }
            dimss.push(shape_);
        }
        Ok(OrtTensorAttr {
            dimss,
            dtypes,
            names,
        })
    }

    pub fn load_onnx<P: AsRef<std::path::Path>>(p: P) -> Result<onnx::ModelProto> {
        let f = std::fs::read(p)?;
        onnx::ModelProto::decode(f.as_slice())
            .map_err(|e| anyhow::anyhow!("Failed to decode ONNX model: {}", e))
    }

    pub fn out_shapes(&self) -> &Vec<Vec<usize>> {
        &self.outputs_attrs.dimss
    }

    pub fn out_dimss(&self) -> &Vec<Vec<usize>> {
        &self.outputs_attrs.dimss
    }

    pub fn out_names(&self) -> &Vec<String> {
        &self.outputs_attrs.names
    }

    pub fn out_d_types(&self) -> &Vec<TensorElementType> {
        &self.outputs_attrs.dtypes
    }

    pub fn in_shapes(&self) -> &Vec<Vec<usize>> {
        &self.inputs_attrs.dimss
    }

    pub fn in_dimss(&self) -> &Vec<Vec<usize>> {
        &self.inputs_attrs.dimss
    }

    pub fn in_names(&self) -> &Vec<String> {
        &self.inputs_attrs.names
    }

    pub fn in_dtypes(&self) -> &Vec<TensorElementType> {
        &self.inputs_attrs.dtypes
    }

    pub fn device(&self) -> &InferenceDevice {
        &self.device
    }

    pub fn inputs_min_opt_max(&self) -> &Vec<Vec<MinOptMax>> {
        &self.inputs_min_opt_max
    }

    pub fn batch(&self) -> &MinOptMax {
        &self.inputs_min_opt_max[0][0]
    }

    pub fn model_width(&self) -> u32 { self.model_width }

    pub fn model_height(&self) -> u32 { self.model_height }

    pub fn try_height(&self) -> Option<&MinOptMax> {
        self.inputs_min_opt_max.first().and_then(|x| x.get(2))
    }

    pub fn try_width(&self) -> Option<&MinOptMax> {
        self.inputs_min_opt_max.first().and_then(|x| x.get(3))
    }

    pub fn height(&self) -> &MinOptMax {
        &self.inputs_min_opt_max[0][2]
    }

    pub fn width(&self) -> &MinOptMax {
        &self.inputs_min_opt_max[0][3]
    }

    pub fn is_batch_dyn(&self) -> bool {
        self.in_shapes()[0][0] == 0
    }

    pub fn try_fetch(&self, key: &str) -> Option<String> {
        match self.session.metadata() {
            Err(_) => None,
            Ok(metadata) => metadata.custom(key).unwrap_or_default(),
        }
    }

    pub fn session(&self) -> &Session {
        &self.session
    }

    pub fn ir_version(&self) -> usize {
        self.model_proto.ir_version as usize
    }

    pub fn opset_version(&self) -> usize {
        self.model_proto.opset_import[0].version as usize
    }

    pub fn producer_name(&self) -> String {
        self.model_proto.producer_name.to_string()
    }

    pub fn producer_version(&self) -> String {
        self.model_proto.producer_version.to_string()
    }

    pub fn model_version(&self) -> usize {
        self.model_proto.model_version as usize
    }

    pub fn parameters(&self) -> usize {
        self.params
    }

    pub fn memory_weights(&self) -> usize {
        self.wb_mem
    }

    pub fn ts(&self) -> &TimeCalc {
        &self.infer_time
    }
}
