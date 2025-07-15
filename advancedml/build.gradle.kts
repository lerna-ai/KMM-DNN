import org.jetbrains.kotlin.gradle.tasks.KotlinCompile

plugins {
	kotlin("multiplatform")
	kotlin("plugin.serialization")
	id("com.android.library")
	id("maven-publish")
}

group = "com.kotlinnlp.simplednn"
version = "0.0.1"

tasks {
	withType<KotlinCompile> {
		kotlinOptions.jvmTarget = "17"
	}
}
kotlin {
	androidTarget {
		publishLibraryVariants("release", "debug")
	}

	listOf(
		iosX64(),
		iosArm64(),
		iosSimulatorArm64()
	).forEach {
		it.binaries.framework {
			baseName = "shared"
			embedBitcode("disable")
		}
	}
	jvm()
	sourceSets {
		val commonMain by getting {
			dependencies {
				implementation(libs.multik.core)
				implementation(libs.multik.kotlin)
				implementation(libs.ktor.client.core)
				implementation(libs.ktor.client.content.negotiation)
				implementation(libs.ktor.serialization.kotlinx.json)
				implementation(libs.ktor.client.cio)
				implementation(libs.ktor.network)
				implementation(libs.ktor.network.tls)
				implementation(libs.kotlin.stdlib.common)
				implementation(libs.kotlinx.serialization.json)
				implementation(libs.kotlinx.coroutines.core)
				implementation(libs.napier)
				implementation(libs.korio)
				runtimeOnly(libs.ktor.utils)
			}
		}
		val commonTest by getting {
			dependencies {
				implementation(kotlin("test"))
			}
		}
		val androidMain by getting {
			dependencies {
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
		val iosX64Main by getting
		val iosArm64Main by getting
		val iosSimulatorArm64Main by getting
		val iosMain by creating {
			dependsOn(commonMain)
			iosX64Main.dependsOn(this)
			iosArm64Main.dependsOn(this)
			iosSimulatorArm64Main.dependsOn(this)
			dependencies {
				implementation(libs.ktor.client.darwin)
			}
		}
		val iosX64Test by getting
		val iosArm64Test by getting
		val iosSimulatorArm64Test by getting
		val iosTest by creating {
			dependsOn(commonTest)
			iosX64Test.dependsOn(this)
			iosArm64Test.dependsOn(this)
			iosSimulatorArm64Test.dependsOn(this)
		}
	}
}
java {
	sourceCompatibility = JavaVersion.VERSION_17
	targetCompatibility = JavaVersion.VERSION_17
}
android {
	testOptions.unitTests.isIncludeAndroidResources = true
	namespace = "com.kotlinnlp.simplednn"
	compileSdk = 33
	defaultConfig {
		minSdk = 23
	}
	compileOptions {
		targetCompatibility = JavaVersion.VERSION_17
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
