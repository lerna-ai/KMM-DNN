plugins {
    `java-library`
}

dependencies {
    api("org.jetbrains.kotlin:kotlin-stdlib-jdk8:1.3.31")
    api("org.jetbrains.kotlin:kotlin-reflect:1.3.31")
    api("org.jblas:jblas:1.2.4")
    api("org.slf4j:slf4j-simple:1.7.21")
    api("com.nhaarman:mockito-kotlin:1.5.0")
    api("org.spekframework.spek2:spek-dsl-jvm:2.0.5")
    api("org.spekframework.spek2:spek-runner-junit5:2.0.5")
    api("com.kotlinnlp:utils:2.1.4")
    testImplementation("org.jetbrains.kotlin:kotlin-test:1.3.31")
}

group = "com.kotlinnlp.simplednn"
version = "0.0.1"
description = "com.kotlinnlp:simplednn"
java.sourceCompatibility = JavaVersion.VERSION_1_8

java {
    withSourcesJar()
}

tasks.withType<JavaCompile>() {
    options.encoding = "UTF-8"
}

tasks.withType<Javadoc>() {
    options.encoding = "UTF-8"
}
