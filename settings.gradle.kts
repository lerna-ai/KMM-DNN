enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
pluginManagement {
	repositories {
		google()
		gradlePluginPortal()
		mavenCentral()
	}
	buildscript {
		repositories {
			mavenCentral()
			maven {
				url = uri("https://storage.googleapis.com/r8-releases/raw")
			}
		}
		dependencies {
			classpath("com.android.tools:r8:8.2.24")
		}
	}
}

dependencyResolutionManagement {
	repositories {
		google()
		mavenCentral()
	}
}

rootProject.name = "KotlinDNN"
include(":advancedml")
