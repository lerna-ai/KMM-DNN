enableFeaturePreview("TYPESAFE_PROJECT_ACCESSORS")
pluginManagement {
	repositories {
		google()
		gradlePluginPortal()
		mavenCentral()
		mavenLocal()
		maven {
			url = uri("https://jcenter.bintray.com/")
		}

		maven {
			url = uri("https://jitpack.io")
		}

		maven {
			url = uri("https://repo.maven.apache.org/maven2/")
		}
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
