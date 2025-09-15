use hound::{SampleFormat, WavReader};
use std::{fs::File, io::Write, path::Path, time::Instant};
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

const MODEL_PATH: &str = "models/ggml-large-v3.bin";

fn main() {
    let audio_path = "audio_en.wav";
    let output_path = "transcription_en.txt";

    // 1. Check files
    if !Path::new(MODEL_PATH).exists() {
        panic!("Model {} not found! Download it with:\nwget https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin -O {}", MODEL_PATH, MODEL_PATH);
    }

    if !Path::new(audio_path).exists() {
        panic!("Audio file {} not found!", audio_path);
    }

    // 2. Load model
    println!("[1/4] Loading model...");
    let ctx = WhisperContext::new_with_params(MODEL_PATH, WhisperContextParameters::default())
        .expect("Error loading model");

    // 3. Load and check audio
    println!("[2/4] Analyzing audio...");
    let audio_data = load_audio(audio_path);

    // 4. Setup parameters for English
    let mut params = FullParams::new(SamplingStrategy::BeamSearch {
        beam_size: 5,
        patience: 1.5,
    });
    params.set_language(Some("en"));
    params.set_translate(false);
    params.set_suppress_blank(true);
    params.set_suppress_nst(true);
    params.set_token_timestamps(true);

    // 5. Transcription
    println!("[3/4] Transcribing...");
    let start_time = Instant::now();
    let mut state = ctx.create_state().expect("Error creating state");
    state
        .full(params, &audio_data)
        .expect("Error during transcription");

    // 6. Save results
    println!("[4/4] Saving...");
    save_results(&state, output_path, start_time);
    println!("Done! Results saved to {}", output_path);
}

fn load_audio(path: &str) -> Vec<f32> {
    let reader = WavReader::open(path).expect("Error reading WAV file");
    let spec = reader.spec();

    // Check format
    if spec.channels != 1 || spec.sample_rate != 16000 {
        panic!("Audio must be mono 16kHz. Convert with:\nffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le audio_en.wav");
    }

    reader
        .into_samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn save_results(state: &whisper_rs::WhisperState, path: &str, start_time: Instant) {
    let mut file = File::create(path).expect("Error creating file");
    writeln!(file, "Transcription results:").unwrap();

    let num_segments = state.full_n_segments().expect("Error getting segments");
    for i in 0..num_segments {
        let text = state.full_get_segment_text(i).expect("Error getting text");
        let start = state.full_get_segment_t0(i).unwrap() as f64 / 100.0;
        let end = state.full_get_segment_t1(i).unwrap() as f64 / 100.0;

        writeln!(file, "[{:.2}s-{:.2}s] {}", start, end, text.trim()).unwrap();
    }

    writeln!(
        file,
        "\nProcessing time: {:.2} sec",
        start_time.elapsed().as_secs_f32()
    )
    .unwrap();
}
