package com.heungbuja.common.config;

import com.heungbuja.admin.entity.Admin;
import com.heungbuja.admin.entity.AdminRole;
import com.heungbuja.admin.repository.AdminRepository;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.boot.CommandLineRunner;
import org.springframework.security.crypto.password.PasswordEncoder;
import org.springframework.stereotype.Component;

@Slf4j
@Component
@RequiredArgsConstructor
public class DataInitializer implements CommandLineRunner {

    private final AdminRepository adminRepository;
    private final PasswordEncoder passwordEncoder;

    @Value("${superadmin.username:superadmin}")
    private String superadminUsername;

    @Value("${superadmin.password:}")
    private String superadminPassword;

    @Override
    public void run(String... args) {
        // 환경변수로 비밀번호가 설정되지 않으면 생성 스킵
        if (superadminPassword == null || superadminPassword.isBlank()) {
            log.warn("=================================================");
            log.warn("SUPERADMIN_PASSWORD 환경변수가 설정되지 않았습니다.");
            log.warn("SuperAdmin 계정이 생성되지 않습니다.");
            log.warn("=================================================");
            return;
        }

        // SUPER_ADMIN 계정이 없으면 생성
        if (!adminRepository.existsByUsername(superadminUsername)) {
            Admin superAdmin = Admin.builder()
                    .username(superadminUsername)
                    .password(passwordEncoder.encode(superadminPassword))
                    .facilityName("흥부자 시스템 관리")
                    .contact("000-0000-0000")
                    .email("admin@heungbuja.com")
                    .role(AdminRole.SUPER_ADMIN)
                    .build();

            adminRepository.save(superAdmin);
            log.info("=================================================");
            log.info("SUPER_ADMIN 계정이 생성되었습니다.");
            log.info("Username: {}", superadminUsername);
            log.info("=================================================");
        } else {
            log.info("SUPER_ADMIN 계정이 이미 존재합니다: {}", superadminUsername);
        }
    }
}
