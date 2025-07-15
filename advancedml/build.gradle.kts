import org.jetbrains.kotlin.gradle.ExperimentalWasmDsl

plugins {
    alias(libs.plugins.kotlinMultiplatform)
    alias(libs.plugins.androidLibrary)
    alias(libs.plugins.kotlinSerialization)

    id("maven-publish")
}

/*tasks{
    withType<Kotlin2JsCompile> {
        kotlinOptions {
            freeCompilerArgs += listOf("-Xskip-prerelease-check")
        }
    }
    withType<KotlinCompile> {
        kotlinOptions.jvmTarget = "21"
    }
}*/

group = "com.kotlinnlp.simplednn"
version = "0.0.1"

kotlin {

    tasks.register("testClasses")

    androidTarget {
        publishLibraryVariants("release", "debug")
        withSourcesJar(publish = false)
    }

    @OptIn(ExperimentalWasmDsl::class)
    wasmJs {
        outputModuleName = "composeDNN"
        browser {
            commonWebpackConfig {
                outputFileName = "composeDNN.js"
            }
        }
        binaries.executable()
    }
    listOf(
        iosX64(),
        iosArm64(),
        iosSimulatorArm64()
//        tvosArm64(),
//        tvosX64(),
//        tvosSimulatorArm64()
    ).forEach {
        it.binaries.framework {
            baseName = "composeDNN"
            //embedBitcode("disable")
        }
    }
    jvm {
         testRuns["test"].executionTask.configure{
            useJUnitPlatform {
                includeEngines("spek2")
            }
        }
    }
    applyDefaultHierarchyTemplate()
    sourceSets {
//	all {
//        	languageSettings.optIn("kotlinx.RequiresOptIn")
//        	languageSettings.optIn("kotlinx.cinterop.ExperimentalForeignApi")
//    	}
        val commonMain by getting {
            dependencies {
                implementation(libs.multik.core)
                implementation(libs.multik.kotlin)
                implementation(libs.ktor.client.core)
                implementation(libs.ktor.client.content.negotiation)
                implementation(libs.ktor.serialization.kotlinx.json)
                implementation(libs.kotlin.stdlib.common)
                implementation(libs.kotlinx.serialization.json)
                implementation(libs.kotlinx.coroutines.core)
                implementation(libs.napier)
                implementation(libs.korge)
                runtimeOnly(libs.ktor.utils)
            }
        }
        val commonTest by getting {
            dependencies {
                implementation(libs.kotlinx.coroutines.test)
                implementation(kotlin("test"))
            }
        }
        val jvmMain by getting
        val jvmTest by getting {
            dependencies {
                implementation(libs.junit)
                implementation(libs.spek.dsl.jvm)
                implementation(libs.spek.runner.junit5)
                implementation(libs.mockito.kotlin)
            }
        }
        val wasmJsMain by getting {
            dependencies {
                implementation(libs.kotlinx.browser)
            }
        }
        val wasmJsTest by getting {
            dependencies {
                implementation(libs.kotlinx.browser)
            }
        }
        val androidMain by getting {
            dependencies {
                implementation(libs.androidx.runtime)
                implementation(libs.ktor.client.cio)
                implementation(libs.ktor.network)
                implementation(libs.ktor.network.tls)
                implementation(libs.ktor.client.okhttp)
                implementation(libs.androidx.work.runtime.ktx)
                implementation(libs.androidx.concurrent.futures.ktx)
            }
        }
        val androidUnitTest by getting {
            dependencies {
                implementation(libs.junit)
                implementation(libs.androidx.core)
                implementation(libs.androidx.junit)
                implementation(libs.robolectric)
                implementation(libs.testng)
                implementation(libs.kotlinx.coroutines.android)
            }
        }
        val nativeMain by getting {
            dependencies {
                implementation(libs.ktor.client.cio)
                implementation(libs.ktor.network)
                implementation(libs.ktor.network.tls)
                implementation(libs.ktor.client.darwin)
            }
        }
        val appleMain by getting
        val iosX64Main by getting
        val iosArm64Main by getting
        val iosSimulatorArm64Main by getting
        val iosMain by getting
        val iosX64Test by getting
        val iosArm64Test by getting
        val iosSimulatorArm64Test by getting
        val iosTest by getting
    }
}

android {
    testOptions.unitTests.isIncludeAndroidResources = true
    namespace = "com.kotlinnlp.simplednn"
    compileSdk = 35
    defaultConfig {
        minSdk = 23
    }
    compileOptions {
        targetCompatibility = JavaVersion.VERSION_21
    }
    buildTypes {
        getByName("release") {
            isMinifyEnabled = true
            proguardFiles(
                "proguard-rules.pro"
            )
            //    consumerProguardFiles("proguard-rules.pro")
        }
        getByName("debug") {
            isMinifyEnabled = false
        }
    }
}

java {
    sourceCompatibility = JavaVersion.VERSION_21
    targetCompatibility = JavaVersion.VERSION_21
}


