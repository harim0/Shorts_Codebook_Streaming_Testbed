plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.shorts.sr"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.shorts.sr"
        minSdk = 26          // API 26: ImageDecoder, ThermalManager baseline
        targetSdk = 34
        versionCode = 1
        versionName = "1.0-phase1"
    }

    buildTypes {
        release {
            isMinifyEnabled = false
        }
    }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        viewBinding = true
    }
}

dependencies {
    // ---- Media3 / ExoPlayer ----
    val media3Version = "1.3.1"
    implementation("androidx.media3:media3-exoplayer:$media3Version")
    implementation("androidx.media3:media3-exoplayer-dash:$media3Version")   // DASH 지원
    implementation("androidx.media3:media3-ui:$media3Version")               // PlayerView
    implementation("androidx.media3:media3-datasource-okhttp:$media3Version") // OkHttp transport

    // ---- Networking (side asset fetch) ----
    implementation("com.squareup.okhttp3:okhttp:4.12.0")

    // ---- Coroutines ----
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.8.0")

    // ---- AndroidX ----
    implementation("androidx.appcompat:appcompat:1.7.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")

    // ---- Phase 2: PyTorch Mobile Lite (주석 처리, Phase 2에서 활성화) ----
    // implementation("org.pytorch:pytorch_android_lite:2.1.0")
}
